from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import torch

from config import (
    DataConfig,
    GlobalTrainingConfig,
    ModelConfig,
    RunConfig,
    TokenConfig,
    TokenizationConfig,
)
from models import TokenChunkDataset, load_model
from plot import plot_training_curves, plot_validation_curves
from tokenizer import build_token_memmap, build_tokenizer, count_tokens
from training import train_model
from utils import generate_text, save_json

SAMPLE_PROMPTS = (
    "Early one morning",
    "One day, a group of friends went to the park.",
    "Tom and his best friend Brian were playing with their cars.",
    "A cat named Mia wanted to go outside.",
)

OUTPUT_SETTINGS = (
    {"temperature": 0.7, "top_k": 30},
    {"temperature": 0.5, "top_k": 10},
    {"temperature": 0.8, "top_k": 50},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fun.ipynb training workflow as a single Python script."
    )
    parser.add_argument("--run-id", help="Artifact run directory name.")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Device to use for training.",
    )
    parser.add_argument(
        "--generation-device",
        choices=("auto", "cuda", "cpu"),
        default="cpu",
        help="Device to use for optional sample generation.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if metrics for this run already exist.",
    )
    parser.add_argument(
        "--generate-samples",
        action="store_true",
        help="Generate sample stories after training finishes.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA mixed precision even when a GPU is available.",
    )
    parser.add_argument(
        "--sample-max-new-tokens",
        type=int,
        default=512,
        help="Max tokens to generate per prompt when --generate-samples is set.",
    )
    parser.add_argument(
        "--max-train-stories",
        type=int,
        default=1_000_000,
        help="Maximum number of stories to read when building tokenizer and memmaps.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=5_000,
        help="Tokenizer vocabulary size.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=512,
        help="Transformer context length.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10_000,
        help="Validation interval in optimizer steps.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2.5e-4,
        help="Peak learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1_000,
        help="Number of warmup steps.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500_000,
        help="Number of training steps.",
    )
    parser.add_argument(
        "--model-name",
        default="xlarge-plus",
        help="Checkpoint name for the trained model.",
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no GPU is available.")
    return torch.device(device_name)


def default_run_id(args: argparse.Namespace) -> str:
    return (
        f"fun-{args.model_name}-steps={args.max_steps}_bs={args.batch_size}"
        f"_ctx={args.context_length}"
    )


def build_configs(
    args: argparse.Namespace,
    train_device: torch.device,
) -> tuple[
    TokenConfig, DataConfig, GlobalTrainingConfig, TokenizationConfig, ModelConfig
]:
    token_config = TokenConfig()
    data_config = DataConfig()
    global_training_config = GlobalTrainingConfig(
        context_length=args.context_length,
        checkpoint_every=args.checkpoint_every,
    )
    tokenization_config = TokenizationConfig(
        vocab_size=args.vocab_size,
        max_train_stories=args.max_train_stories,
    )
    model_config = ModelConfig(
        name=args.model_name,
        d_model=768,
        n_heads=12,
        n_layers=6,
        d_ff=3072,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        use_amp=(not args.no_amp) and train_device.type == "cuda",
    )
    return (
        token_config,
        data_config,
        global_training_config,
        tokenization_config,
        model_config,
    )


def prepare_datasets(
    data_config: DataConfig,
    global_training_config: GlobalTrainingConfig,
    token_config: TokenConfig,
    tokenization_config: TokenizationConfig,
):
    tokenizer = build_tokenizer(
        tokenization_config,
        token_config,
        data_config.training_file,
    )

    train_token_count = count_tokens(
        tokenization_config,
        tokenizer,
        data_config.training_file,
    )
    valid_token_count = count_tokens(
        tokenization_config,
        tokenizer,
        data_config.validation_file,
    )
    print(
        f"Counted {train_token_count:,} train tokens and {valid_token_count:,} validation tokens"
    )

    train_token_memmap_path = build_token_memmap(
        tokenization_config,
        token_config,
        tokenizer,
        data_config.training_file,
        train_token_count,
    )
    valid_token_memmap_path = build_token_memmap(
        tokenization_config,
        token_config,
        tokenizer,
        data_config.validation_file,
        valid_token_count,
    )

    train_dataset = TokenChunkDataset(
        train_token_memmap_path,
        train_token_count,
        global_training_config.context_length,
    )
    valid_dataset = TokenChunkDataset(
        valid_token_memmap_path,
        valid_token_count,
        global_training_config.context_length,
    )
    return tokenizer, train_dataset, valid_dataset


def generate_samples(
    run_config: RunConfig,
    model_path: str,
    token_config: TokenConfig,
    global_training_config: GlobalTrainingConfig,
    generation_device: torch.device,
    tokenizer,
    max_new_tokens: int,
) -> tuple[list[dict], str]:
    model = load_model(model_path, generation_device)
    generations: list[dict] = []
    for prompt in SAMPLE_PROMPTS:
        for setting in OUTPUT_SETTINGS:
            generation = generate_text(
                token_config,
                global_training_config,
                model,
                tokenizer,
                prompt,
                generation_device,
                max_new_tokens=max_new_tokens,
                temperature=setting["temperature"],
                top_k=setting["top_k"],
            )
            generations.append(
                {
                    "model": model_path,
                    "prompt": prompt,
                    "generated_text": generation,
                    "temperature": setting["temperature"],
                    "top_k": setting["top_k"],
                }
            )

    samples_dir = run_config.run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    generations_path = samples_dir / "generations.json"
    generations_path.write_text(json.dumps(generations, indent=2), encoding="utf-8")
    return generations, str(generations_path)


def save_manifest(
    run_config: RunConfig,
    data_config: DataConfig,
    tokenization_config: TokenizationConfig,
    global_training_config: GlobalTrainingConfig,
    model_config: ModelConfig,
    train_device: torch.device,
    generations_path: str | None,
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
        "data": {
            "training_file": str(data_config.training_file),
            "validation_file": str(data_config.validation_file),
        },
        "tokenization": asdict(tokenization_config),
        "global_training": asdict(global_training_config),
        "model": asdict(model_config),
        "artifacts": {
            "models_dir": str(run_config.models),
            "metrics_dir": str(run_config.metrics),
            "plots_dir": str(run_config.plots),
            "generations": generations_path,
        },
        "generation_settings": list(OUTPUT_SETTINGS) if generations_path else None,
    }
    manifest_path = run_config.run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved manifest to {manifest_path}")


def main() -> None:
    args = parse_args()
    train_device = resolve_device(args.device)
    generation_device = (
        resolve_device(args.generation_device) if args.generate_samples else None
    )

    if train_device.type == "cuda":
        print(f"Training on CUDA: {torch.cuda.get_device_name(0)}")
        torch.set_float32_matmul_precision("high")
    else:
        print("Training on CPU")

    run_id = args.run_id or default_run_id(args)
    run_config = RunConfig(run_id)

    (
        token_config,
        data_config,
        global_training_config,
        tokenization_config,
        model_config,
    ) = build_configs(args, train_device)

    tokenizer, train_dataset, valid_dataset = prepare_datasets(
        data_config,
        global_training_config,
        token_config,
        tokenization_config,
    )

    metrics_path = run_config.metrics / f"{model_config.name}.json"
    if metrics_path.exists() and not args.force:
        print(f"Found existing metrics at {metrics_path}, skipping training")
        result = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        result = train_model(
            run_config,
            global_training_config,
            model_config,
            tokenizer,
            train_dataset,
            valid_dataset,
            device=train_device,
        )
        save_json(result, metrics_path)
        print(f"Saved metrics to {metrics_path}")

    plot_training_curves(
        results={model_config.name: result},
        output_path=run_config.plots / "training_loss.png",
    )
    plot_validation_curves(
        results={model_config.name: result},
        output_path=run_config.plots / "validation_loss.png",
    )
    print(f"Saved plots to {run_config.plots}")

    generations_path = None
    if args.generate_samples:
        if train_device.type == "cuda":
            torch.cuda.empty_cache()
        _, generations_path = generate_samples(
            run_config,
            result["model_path"],
            token_config,
            global_training_config,
            generation_device,
            tokenizer,
            args.sample_max_new_tokens,
        )
        print(f"Saved generations to {generations_path}")

    save_manifest(
        run_config,
        data_config,
        tokenization_config,
        global_training_config,
        model_config,
        train_device,
        generations_path,
    )
    print(f"Finished run {run_id}")


if __name__ == "__main__":
    main()
