import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from helpers import apply_lora_to_model, freeze_non_lora_parameters

from config import (
    DataConfig,
    GlobalTrainingConfig,
    ModelConfig,
    RunConfig,
    TokenConfig,
    TokenizationConfig,
)
from models import TokenChunkDataset, load_model
from tokenizer import build_token_memmap, build_tokenizer, count_tokens
from training import compute_lr
from utils import count_parameters, make_dataloader, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune a pretrained TinyGPT checkpoint on instructions."
    )
    parser.add_argument(
        "pretrained_model",
        type=Path,
        help="Path to the pretrained TinyGPT checkpoint.",
    )
    parser.add_argument(
        "--instruction-training-file",
        type=Path,
        help="Path to the instruction dataset text file for training.",
        required=True,
    )
    parser.add_argument(
        "--instruction-validation-file",
        type=Path,
        help="Path to the instruction dataset text file for validation.",
        required=True,
    )
    parser.add_argument(
        "--run-id",
        default="fine-tune-lora",
        help="Artifact run directory name.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Device to use for training.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Peak learning rate for LoRA parameters.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=50,
        help="Number of warmup steps.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2_000,
        help="Number of optimizer steps.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Logging interval in optimizer steps.",
    )
    parser.add_argument(
        "--max-train-stories",
        type=int,
        default=1_000_000,
        help="Tokenizer cache key used in the original pretraining run.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=8,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=16.0,
        help="LoRA scaling factor.",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="Dropout used on the LoRA branch.",
    )
    parser.add_argument(
        "--target-ff",
        action="store_true",
        help="Also apply LoRA to the feedforward linear layers.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA mixed precision even when a GPU is available.",
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no GPU is available.")
    return torch.device(device_name)


def load_checkpoint_configs(
    checkpoint_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[
    TokenConfig, DataConfig, GlobalTrainingConfig, TokenizationConfig, ModelConfig
]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_model_config = ModelConfig(**checkpoint["config"])

    token_config = TokenConfig()
    data_config = DataConfig(
        instruction_training_file=args.instruction_training_file,
        instruction_validation_file=args.instruction_validation_file,
    )
    global_training_config = GlobalTrainingConfig(
        context_length=checkpoint["context_length"],
        checkpoint_every=args.checkpoint_every,
    )
    tokenization_config = TokenizationConfig(
        vocab_size=checkpoint["vocab_size"],
        max_train_stories=args.max_train_stories,
    )
    fine_tune_model_config = ModelConfig(
        name=f"{checkpoint_model_config.name}-lora",
        d_model=checkpoint_model_config.d_model,
        n_heads=checkpoint_model_config.n_heads,
        n_layers=checkpoint_model_config.n_layers,
        d_ff=checkpoint_model_config.d_ff,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        dropout=checkpoint_model_config.dropout,
        grad_clip_norm=checkpoint_model_config.grad_clip_norm,
        use_amp=(not args.no_amp) and device.type == "cuda",
    )
    return (
        token_config,
        data_config,
        global_training_config,
        tokenization_config,
        fine_tune_model_config,
    )


def prepare_instruction_dataset(
    instruction_file: Path,
    token_config: TokenConfig,
    tokenization_config: TokenizationConfig,
    context_length: int,
    tokenizer_source_path: Path,
) -> tuple:
    tokenizer = build_tokenizer(
        tokenization_config,
        token_config,
        tokenizer_source_path,
    )
    token_count = count_tokens(
        tokenization_config,
        tokenizer,
        instruction_file,
    )
    token_memmap_path = build_token_memmap(
        tokenization_config,
        token_config,
        tokenizer,
        instruction_file,
        token_count,
    )
    dataset = TokenChunkDataset(
        token_memmap_path,
        token_count,
        context_length,
    )
    return tokenizer, dataset, token_count, token_memmap_path


def _find_subsequence_positions(sequence: list[int], pattern: list[int]) -> list[int]:
    if not pattern or len(pattern) > len(sequence):
        return []
    positions: list[int] = []
    width = len(pattern)
    for index in range(len(sequence) - width + 1):
        if sequence[index : index + width] == pattern:
            positions.append(index)
    return positions


def mask_response_only_targets(
    x: torch.Tensor,
    y: torch.Tensor,
    prompt_pattern: list[int],
    response_pattern: list[int],
) -> torch.Tensor:
    masked_y = y.clone()

    for batch_index in range(x.size(0)):
        x_row = x[batch_index].tolist()
        prompt_positions = _find_subsequence_positions(x_row, prompt_pattern)
        response_positions = _find_subsequence_positions(x_row, response_pattern)

        if not prompt_positions and not response_positions:
            continue

        prompt_starts = set(prompt_positions)
        response_starts = set(response_positions)

        in_response = False
        for token_index in range(x.size(1)):
            if token_index in prompt_starts:
                in_response = False
            if token_index in response_starts:
                in_response = True

            if not in_response:
                masked_y[batch_index, token_index] = -100

    return masked_y


def save_manifest(
    run_config: RunConfig,
    data_config: DataConfig,
    tokenization_config: TokenizationConfig,
    global_training_config: GlobalTrainingConfig,
    model_config: ModelConfig,
    train_device: torch.device,
    args: argparse.Namespace,
    model_path: Path,
    metrics_path: Path,
    train_memmap_path: Path,
    valid_memmap_path: Path,
) -> None:
    manifest = {
        "run_id": run_config.run_id,
        "run_dir": str(run_config.run_dir),
        "device": str(train_device),
        "cuda_device": (
            torch.cuda.get_device_name(0)
            if train_device.type == "cuda" and torch.cuda.is_available()
            else None
        ),
        "base_checkpoint": str(args.pretrained_model),
        "data": {
            "training_file": str(data_config.instruction_training_file),
            "validation_file": str(data_config.instruction_validation_file),
        },
        "tokenization": asdict(tokenization_config),
        "global_training": asdict(global_training_config),
        "model": asdict(model_config),
        "lora": {
            "rank": args.rank,
            "alpha": args.alpha,
            "dropout": args.lora_dropout,
            "target_ff": args.target_ff,
        },
        "artifacts": {
            "models_dir": str(run_config.models),
            "metrics_dir": str(run_config.metrics),
            "plots_dir": str(run_config.plots),
            "model_path": str(model_path),
            "metrics_path": str(metrics_path),
            "instruction_train_memmap_path": str(train_memmap_path),
            "instruction_valid_memmap_path": str(valid_memmap_path),
        },
    }
    manifest_path = run_config.run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved manifest to {manifest_path}")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    (
        token_config,
        data_config,
        global_training_config,
        tokenization_config,
        model_config,
    ) = load_checkpoint_configs(args.pretrained_model, args, device)

    tokenizer, train_dataset, train_token_count, train_memmap_path = (
        prepare_instruction_dataset(
            args.instruction_training_file,
            token_config,
            tokenization_config,
            global_training_config.context_length,
            Path(data_config.training_file),
        )
    )
    _, valid_dataset, valid_token_count, valid_memmap_path = (
        prepare_instruction_dataset(
            args.instruction_validation_file,
            token_config,
            tokenization_config,
            global_training_config.context_length,
            Path(data_config.training_file),
        )
    )

    train_loader = make_dataloader(
        train_dataset,
        batch_size=model_config.batch_size,
        shuffle=True,
    )
    valid_loader = make_dataloader(
        valid_dataset,
        batch_size=model_config.batch_size,
        shuffle=False,
    )

    model = load_model(args.pretrained_model, device)
    model.train()
    model = apply_lora_to_model(
        model,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.lora_dropout,
        target_ff=args.target_ff,
    )
    model = model.to(device)
    freeze_non_lora_parameters(model)

    trainable_parameters = [
        param for param in model.parameters() if param.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=model_config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=model_config.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=model_config.use_amp and device.type == "cuda",
    )

    run_config = RunConfig(args.run_id)
    train_iter = iter(train_loader)
    train_history: list[dict] = []
    valid_history: list[dict] = []
    start_time = time.perf_counter()
    total_tokens_processed = 0
    autocast_dtype = torch.float16 if device.type == "cuda" else None
    vocab_size = tokenizer.get_vocab_size()
    prompt_pattern = tokenizer.encode("Prompt:").ids
    response_pattern = tokenizer.encode("Response:").ids

    for step in range(1, model_config.max_steps + 1):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        lr = compute_lr(
            step,
            model_config.max_steps,
            model_config.warmup_steps,
            model_config.learning_rate,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        masked_y = mask_response_only_targets(x, y, prompt_pattern, response_pattern)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=model_config.use_amp and device.type == "cuda",
        ):
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                masked_y.view(-1),
                ignore_index=-100,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                model_config.grad_clip_norm,
            )
        )
        scaler.step(optimizer)
        scaler.update()

        batch_tokens = int(x.numel())
        total_tokens_processed += batch_tokens
        elapsed = time.perf_counter() - start_time
        tokens_per_second = total_tokens_processed / max(elapsed, 1e-9)
        train_history.append(
            {
                "step": step,
                "loss": float(loss.item()),
                "perplexity": float(math.exp(loss.item())),
                "lr": lr,
                "grad_norm": grad_norm,
                "tokens_per_second": tokens_per_second,
                "elapsed_seconds": elapsed,
            }
        )

        if step == 1 or step % global_training_config.checkpoint_every == 0:
            model.eval()
            losses = []
            total_tokens = 0
            with torch.no_grad():
                for valid_x, valid_y in valid_loader:
                    valid_x = valid_x.to(device, non_blocking=True)
                    valid_y = valid_y.to(device, non_blocking=True)
                    masked_valid_y = mask_response_only_targets(
                        valid_x,
                        valid_y,
                        prompt_pattern,
                        response_pattern,
                    )
                    with torch.autocast(
                        device_type=device.type,
                        dtype=autocast_dtype,
                        enabled=model_config.use_amp and device.type == "cuda",
                    ):
                        valid_logits = model(valid_x)
                        valid_loss = F.cross_entropy(
                            valid_logits.view(-1, vocab_size),
                            masked_valid_y.view(-1),
                            ignore_index=-100,
                        )
                    losses.append(valid_loss.item())
                    total_tokens += int((masked_valid_y != -100).sum().item())
            model.train()
            metrics = {
                "loss": float(sum(losses) / max(1, len(losses))),
                "perplexity": float(math.exp(sum(losses) / max(1, len(losses)))),
                "tokens_evaluated": total_tokens,
            }
            metrics["step"] = step
            valid_history.append(metrics)
            print(
                f"[{model_config.name}] step={step} "
                f"train_loss={loss.item():.4f} "
                f"train_ppl={math.exp(loss.item()):.3f} "
                f"valid_loss={metrics['loss']:.4f} "
                f"valid_ppl={metrics['perplexity']:.3f}"
            )

    total_time = time.perf_counter() - start_time
    model_path = run_config.models / f"{model_config.name}.pt"
    metrics_path = run_config.metrics / f"{model_config.name}.json"

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(model_config),
            "vocab_size": vocab_size,
            "context_length": global_training_config.context_length,
            "base_checkpoint": str(args.pretrained_model),
            "instruction_training_file": str(args.instruction_training_file),
            "instruction_validation_file": str(args.instruction_validation_file),
            "lora": {
                "rank": args.rank,
                "alpha": args.alpha,
                "dropout": args.lora_dropout,
                "target_ff": args.target_ff,
            },
        },
        model_path,
    )
    save_json(
        {
            "config": asdict(model_config),
            "parameter_count": count_parameters(model),
            "train_history": train_history,
            "valid_history": valid_history,
            "model_path": str(model_path),
            "instruction_training_file": str(args.instruction_training_file),
            "instruction_validation_file": str(args.instruction_validation_file),
            "instruction_token_count": train_token_count,
            "validation_token_count": valid_token_count,
            "instruction_train_memmap_path": str(train_memmap_path),
            "instruction_valid_memmap_path": str(valid_memmap_path),
            "total_training_time_seconds": total_time,
            "tokens_processed": total_tokens_processed,
            "tokens_per_second": total_tokens_processed / max(total_time, 1e-9),
        },
        metrics_path,
    )
    save_manifest(
        run_config,
        data_config,
        tokenization_config,
        global_training_config,
        model_config,
        device,
        args,
        model_path,
        metrics_path,
        train_memmap_path,
        valid_memmap_path,
    )
    print(f"Saved LoRA fine-tuned checkpoint to {model_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
