#!/usr/bin/env pkgx +python@3.13 uv run
"""
Improved LoRA fine-tuning for TinyGPT with proper instruction dataset.

Key improvements over fine_tune.py:
  1. Uses InstructionDataset — keeps prompt/response pairs intact (no chunking)
  2. Higher LoRA rank (32) + alpha (64) for learning semantic distinctions
  3. Higher learning rate (2e-4) for faster LoRA convergence
  4. Response-only loss baked into the dataset labels (no fragile pattern matching)
  5. Cosine LR schedule with warmup
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from helpers import apply_lora_to_model, freeze_non_lora_parameters
from instruction_dataset import InstructionDataset

from config import (
    DataConfig,
    GlobalTrainingConfig,
    ModelConfig,
    RunConfig,
    TokenConfig,
    TokenizationConfig,
)
from models import load_model
from tokenizer import build_tokenizer
from utils import count_parameters, make_dataloader, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Improved LoRA fine-tune for TinyGPT with proper instruction dataset."
    )
    parser.add_argument(
        "pretrained_model",
        type=Path,
        help="Path to pretrained TinyGPT checkpoint.",
    )
    parser.add_argument(
        "--instruction-training-file",
        type=Path,
        required=True,
        help="Path to instruction dataset (Prompt:/Response: pairs).",
    )
    parser.add_argument(
        "--instruction-validation-file",
        type=Path,
        required=True,
        help="Path to instruction validation dataset.",
    )
    parser.add_argument(
        "--run-id",
        default="fine-tune-v2",
        help="Artifact run directory name.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=3_000)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--max-train-stories", type=int, default=1_000_000)
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank.")
    parser.add_argument("--alpha", type=float, default=64.0, help="LoRA scaling factor.")
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-ff",
        action="store_true",
        help="Also apply LoRA to feedforward layers.",
    )
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no GPU is available.")
    return torch.device(device_name)


def compute_lr(step: int, max_steps: int, warmup_steps: int, base_lr: float) -> float:
    if step <= warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    min_lr = 0.1 * base_lr
    return min_lr + (base_lr - min_lr) * cosine


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    vocab_size: int,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    autocast_dtype = torch.float16 if device.type == "cuda" else None

    for input_ids, labels in data_loader:
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=use_amp and device.type == "cuda",
        ):
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        total_loss += loss.item()
        total_tokens += int((labels != -100).sum().item())

    model.train()
    mean_loss = total_loss / max(1, len(data_loader))
    return {
        "loss": mean_loss,
        "perplexity": float(math.exp(mean_loss)),
        "tokens_evaluated": total_tokens,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        print(f"Training on: {torch.cuda.get_device_name(0)}")
    else:
        print("Training on CPU")

    # --- Load checkpoint and build configs ---
    checkpoint = torch.load(args.pretrained_model, map_location=device)
    checkpoint_model_config = ModelConfig(**checkpoint["config"])
    context_length = checkpoint["context_length"]
    vocab_size = checkpoint["vocab_size"]

    token_config = TokenConfig()
    data_config = DataConfig(
        instruction_training_file=args.instruction_training_file,
        instruction_validation_file=args.instruction_validation_file,
    )
    global_training_config = GlobalTrainingConfig(
        context_length=context_length,
        checkpoint_every=args.checkpoint_every,
    )
    tokenization_config = TokenizationConfig(
        vocab_size=vocab_size,
        max_train_stories=args.max_train_stories,
    )
    fine_tune_config = ModelConfig(
        name=f"{checkpoint_model_config.name}-lora-v2",
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

    # --- Build tokenizer and datasets ---
    tokenizer = build_tokenizer(
        tokenization_config,
        token_config,
        Path(data_config.training_file),
    )

    print(f"Loading training data from {args.instruction_training_file} ...")
    train_dataset = InstructionDataset(
        instruction_file=str(args.instruction_training_file),
        tokenizer=tokenizer,
        context_length=context_length,
    )
    print(f"  -> {len(train_dataset)} complete instruction pairs")

    print(f"Loading validation data from {args.instruction_validation_file} ...")
    valid_dataset = InstructionDataset(
        instruction_file=str(args.instruction_validation_file),
        tokenizer=tokenizer,
        context_length=context_length,
    )
    print(f"  -> {len(valid_dataset)} complete instruction pairs")

    train_loader = make_dataloader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    valid_loader = make_dataloader(
        valid_dataset, batch_size=args.batch_size, shuffle=False
    )

    # --- Load model and apply LoRA ---
    model = load_model(args.pretrained_model, device, train=True)
    model = apply_lora_to_model(
        model,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.lora_dropout,
        target_ff=args.target_ff,
    )
    model = model.to(device)
    freeze_non_lora_parameters(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = count_parameters(model)
    trainable_count = sum(p.numel() for p in trainable_params)
    print(
        f"Model: {total_params:,} total params, "
        f"{trainable_count:,} trainable (LoRA rank={args.rank})"
    )

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=fine_tune_config.use_amp and device.type == "cuda",
    )

    # --- Training loop ---
    run_config = RunConfig(args.run_id)
    train_iter = iter(train_loader)
    train_history: list[dict] = []
    valid_history: list[dict] = []
    start_time = time.perf_counter()
    total_tokens_processed = 0
    autocast_dtype = torch.float16 if device.type == "cuda" else None
    best_valid_loss = float("inf")

    for step in range(1, fine_tune_config.max_steps + 1):
        try:
            input_ids, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            input_ids, labels = next(train_iter)

        lr = compute_lr(
            step,
            fine_tune_config.max_steps,
            fine_tune_config.warmup_steps,
            fine_tune_config.learning_rate,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=fine_tune_config.use_amp and device.type == "cuda",
        ):
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                trainable_params, fine_tune_config.grad_clip_norm
            )
        )
        scaler.step(optimizer)
        scaler.update()

        batch_tokens = int((labels != -100).sum().item())
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
            metrics = evaluate(
                model,
                valid_loader,
                device,
                vocab_size=vocab_size,
                use_amp=fine_tune_config.use_amp,
            )
            metrics["step"] = step
            valid_history.append(metrics)

            ppl = metrics["perplexity"]
            best_marker = ""
            if metrics["loss"] < best_valid_loss:
                best_valid_loss = metrics["loss"]
                best_marker = " *BEST*"

            print(
                f"[{fine_tune_config.name}] step={step:>5} "
                f"train_loss={loss.item():.4f} train_ppl={math.exp(loss.item()):.3f} "
                f"valid_loss={metrics['loss']:.4f} valid_ppl={ppl:.3f}"
                f"{best_marker}"
            )

    # --- Save ---
    total_time = time.perf_counter() - start_time
    model_path = run_config.models / f"{fine_tune_config.name}.pt"
    metrics_path = run_config.metrics / f"{fine_tune_config.name}.json"

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(checkpoint_model_config),
            "vocab_size": vocab_size,
            "context_length": context_length,
            "lora": {
                "rank": args.rank,
                "alpha": args.alpha,
                "dropout": args.lora_dropout,
                "target_ff": args.target_ff,
            },
            "base_checkpoint": str(args.pretrained_model),
        },
        model_path,
    )
    print(f"Saved model to {model_path}")

    result = {
        "config": asdict(fine_tune_config),
        "parameter_count": total_params,
        "trainable_parameters": trainable_count,
        "train_history": train_history,
        "valid_history": valid_history,
        "model_path": str(model_path),
        "total_training_time_seconds": total_time,
        "tokens_processed": total_tokens_processed,
        "tokens_per_second": total_tokens_processed / max(total_time, 1e-9),
        "best_valid_loss": best_valid_loss,
    }
    save_json(result, metrics_path)
    print(f"Saved metrics to {metrics_path}")
    print(
        f"Done! {total_time:.0f}s, {total_tokens_processed:,} tokens, "
        f"best valid loss: {best_valid_loss:.4f}"
    )


if __name__ == "__main__":
    main()
