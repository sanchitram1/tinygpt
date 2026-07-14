"""Command-line entry point for creating an immutable serving bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from core.artifacts import create_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a TinyGPT serving bundle")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--model-version")
    args = parser.parse_args()
    output = create_bundle(
        args.checkpoint,
        args.tokenizer,
        args.output,
        run_id=args.run_id,
        model_version=args.model_version,
    )
    print(output)
