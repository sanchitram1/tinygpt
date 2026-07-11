# Training

## Overview

This directory owns pretraining and fine-tuning orchestration. Shared model,
tokenizer, generation, and artifact code belongs in `../core/`.

Read [Plan 1](../plans/01-stabilize-training-pipeline.md) before changing the
training pipeline.

## Standards

- Keep runs reproducible: record config, seed, dataset/tokenizer IDs, and code
  provenance with every produced artifact.
- Preserve `../core/artifacts/baseline/` as read-only. Do not retrain the
  historical baseline unless the task explicitly requests it.
- New training must save latest resumable, best-validation, and final models.
- Support CPU, CUDA, and MPS without calling CUDA-only APIs on other devices.
- Keep one experiment to one primary hypothesis; do not mix data, optimizer,
  architecture, and decoding changes.

## Verification

Run focused tests for changed behavior, then:

```bash
cd training && uv run ruff check .
cd training && uv run pytest
git diff --check
```

Use small CPU/MPS smoke runs for pipeline changes. Do not start long training or
rent GPU infrastructure without explicit approval.
