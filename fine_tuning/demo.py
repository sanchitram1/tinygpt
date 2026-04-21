import argparse
from pathlib import Path

import torch
from helpers import apply_lora_to_model

from config import GlobalTrainingConfig, TokenConfig, TokenizationConfig
from models import TinyGPT
from tokenizer import build_tokenizer
from utils import generate_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run interactive demo prompts against a fine-tuned TinyGPT checkpoint."
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to a fine-tuned checkpoint saved by fine_tune.py.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Device to use for generation.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Optional one-shot prompt. If omitted, starts an interactive REPL.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top-k sampling cutoff.",
    )
    parser.add_argument(
        "--max-train-stories",
        type=int,
        default=1_000_000,
        help="Tokenizer cache key used in the original pretraining run.",
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no GPU is available.")
    return torch.device(device_name)


def load_finetuned_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[TinyGPT, GlobalTrainingConfig, Path]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    model = TinyGPT(
        vocab_size=checkpoint["vocab_size"],
        context_length=checkpoint["context_length"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
    ).to(device)

    lora_config = checkpoint.get("lora")
    if lora_config:
        model = apply_lora_to_model(
            model,
            rank=int(lora_config["rank"]),
            alpha=float(lora_config["alpha"]),
            dropout=float(lora_config["dropout"]),
            target_ff=bool(lora_config.get("target_ff", False)),
        )

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    global_training_config = GlobalTrainingConfig(
        context_length=checkpoint["context_length"],
    )
    base_checkpoint = checkpoint.get("base_checkpoint")
    return (
        model,
        global_training_config,
        Path(base_checkpoint) if base_checkpoint else checkpoint_path,
    )


def build_prompt(user_prompt: str) -> str:
    return f"Prompt: {user_prompt}\nResponse:"


def run_one_prompt(
    model: TinyGPT,
    tokenizer,
    token_config: TokenConfig,
    global_training_config: GlobalTrainingConfig,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    formatted_prompt = build_prompt(prompt)
    output = generate_text(
        token_config,
        global_training_config,
        model,
        tokenizer,
        formatted_prompt,
        device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )

    if "Response:" in output:
        return output.split("Response:", 1)[1].strip()
    return output.strip()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    token_config = TokenConfig()

    model, global_training_config, _ = load_finetuned_model(args.checkpoint, device)
    tokenizer = build_tokenizer(
        TokenizationConfig(
            vocab_size=model.lm_head.out_features,
            max_train_stories=args.max_train_stories,
        ),
        token_config,
        Path(__file__).resolve().parent.parent
        / "data"
        / "TinyStoriesV2-GPT4-train.txt",
    )

    if args.prompt:
        print(
            run_one_prompt(
                model,
                tokenizer,
                token_config,
                global_training_config,
                args.prompt,
                device,
                args.max_new_tokens,
                args.temperature,
                args.top_k,
            )
        )
        return

    print("Interactive TinyStories fine-tuning demo")
    print("Enter a story instruction like: Tell me a tale about a brave little fox.")
    print("Press Ctrl-D or submit an empty line to exit.")

    while True:
        try:
            prompt = input("\nPrompt> ").strip()
        except EOFError:
            print()
            break

        if not prompt:
            break

        response = run_one_prompt(
            model,
            tokenizer,
            token_config,
            global_training_config,
            prompt,
            device,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
        )
        print(f"\nResponse:\n{response}")


if __name__ == "__main__":
    main()
