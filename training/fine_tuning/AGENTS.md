# Fine-Tuning

## Overview

This directory adapts a pretrained TinyGPT checkpoint to instruction-following
children's-story prompts. Read `README.md` before changing the data format.

## Standards

- Keep prompt/response pairs intact and compute loss only on the response when
  using response-only training.
- A fine-tuned run must record its base checkpoint digest, tokenizer ID,
  instruction-dataset digest, and LoRA configuration.
- Write new artifacts to a distinct run directory; never overwrite the base
  checkpoint or baseline artifacts.
- Treat generated prompts as data: preserve their source and avoid silently
  changing a released training split.

## Verification

Use a small instruction dataset and a few steps to validate loading, masking,
checkpointing, and resume behavior before a full run. Run focused tests and
`git diff --check`.
