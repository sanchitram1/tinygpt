# Plan 1: Stabilize the Training Pipeline

## Objective

Make TinyGPT training reproducible, resumable, portable across CPU, CUDA, and
Apple Silicon MPS, and auditable from the recovered baseline through every
future experiment.

This plan does not attempt to recreate missing intermediate checkpoints from
the original baseline training. It first verifies the recovered 5,000-token
tokenizer against the checkpoint, then establishes a new artifact contract so
all future runs are fully recoverable.

## Recovered Inputs

| Asset | Location | Status |
|---|---|---|
| Baseline checkpoint | `artifacts/baseline/xlarge-plus.pt` | Available |
| Baseline metrics | `artifacts/baseline/xlarge-plus.json` | Available |
| Baseline manifest | `artifacts/baseline/manifest.json` | Available, with historical Colab paths |
| Training corpus | `data/TinyStoriesV2-GPT4-train-001.txt` | Available |
| Validation corpus | `data/TinyStoriesV2-GPT4-valid.txt` | Available |
| Fine-tuning data | `data/fine_tuning-instructions-merged-train.txt`, `data/fine_tuning-instructions-templates-valid.txt` | Available |
| Baseline tokenizer | `artifacts/tokenizers/tinystories_bpe_metaspace_5000_1000000.json` | Available; verify vocabulary size and digest before use |
| Intermediate training checkpoints | Not present | Unrecoverable |

The recovered manifest records the `xlarge-plus` baseline as a 46,368,768
parameter model trained for 100,000 steps on an A100. Its historical paths are
informational only; no implementation should depend on them.

## Scope

1. Replace implicit artifact placement with an explicit, portable run layout.
2. Seed all sources of randomness and retain their state in checkpoints.
3. Record hashes for source datasets, generated tokenizer, source code, and
   immutable run configuration.
4. Save both latest resumable and best-validation checkpoints.
5. Resume exactly from a latest checkpoint, including optimizer, scheduler,
   gradient scaler, RNG state, and token counters.
6. Support `cpu`, `cuda`, and `mps` devices with capability-based AMP behavior.
7. Add focused tests and a short smoke-training command.

## Target Artifact Contract

Every new run lives at `artifacts/runs/<run_id>/`:

```text
artifacts/runs/<run_id>/
  manifest.json
  config.json
  provenance.json
  metrics/
    training.jsonl
    validation.jsonl
    summary.json
  checkpoints/
    latest.pt
    best-validation.pt
  models/
    final.pt
  tokenizers/
    tokenizer.json
    tokenizer.sha256
  samples/
    generations.json
  plots/
```

`provenance.json` must include:

- SHA-256 and byte count for each input dataset and generated tokenizer;
- checkpoint parent digest for resumed or fine-tuned runs;
- git commit SHA and dirty-worktree indicator, when git metadata is available;
- Python, PyTorch, tokenizers, and platform versions;
- selected device and device name where available;
- all random seeds and deterministic-algorithm flags;
- stable `run_id`, `model_version`, and `tokenizer_id` values.

`model_version` should be a readable identifier derived from model configuration
and run ID. `tokenizer_id` should include vocabulary configuration plus the
tokenizer SHA-256 prefix. Neither must depend on absolute paths.

## Target Repository Layout

Separate runtime primitives from orchestration. The model, tokenizer, generation,
and artifact-reading code must not live under `training/`, because the evaluator
and FastAPI service need the same tested implementations.

```text
tinygpt/
  core/                 # model, tokenizer, generation, configuration, artifacts
  training/             # pretraining and fine-tuning commands and training loops
  evals/                # versioned cases, local runners, evaluators, reports
  service/              # FastAPI application, static chat UI, telemetry
  tests/
  plans/
```

The migration should retain temporary CLI compatibility wrappers at the
repository root only when needed, then remove them deliberately after callers
and documentation use the new package paths. No module may duplicate core model
or tokenizer logic during the migration.

## Implementation Steps

### 1. Establish an artifact and configuration API

- Refactor `RunConfig` and data-path resolution so they accept a repository
  root/artifact root rather than infer Colab or `/workspace` paths.
- Move existing pretraining and fine-tuning orchestration into `training/` and
  extract reusable model/tokenizer/generation/artifact primitives into `core/`.
- Preserve an explicit override for externally mounted data and artifacts.
- Move all write operations through a small artifact helper. It must create
  missing subdirectories even when resuming an existing run.
- Add a migration/read-only adapter for `artifacts/baseline/` so the baseline
  can be evaluated and sampled without rewriting or misrepresenting its
  historical provenance.

### 2. Make preprocessing identifiable and reproducible

- Add a streaming SHA-256 helper for datasets and tokenizer artifacts.
- Copy the tokenizer used by every run into that run's `tokenizers/` directory;
  load it from there for generation, fine-tuning, and serving.
- Make token memmap cache names incorporate the source-data hash, tokenizer ID,
  context length, and split. Never reuse a cache based only on filename and
  vocabulary size.
- Record counts of stories, tokens, chunks, and dropped/truncated examples.
- Verify that the training and validation source digests differ.

### 3. Add deterministic execution controls

- Add `--seed` to training, generation, and fine-tuning commands.
- Seed Python `random`, NumPy, PyTorch CPU, all CUDA devices, and the
  DataLoader generator. Record every seed in provenance.
- Make sample generation accept a per-prompt/per-decoding seed and store it
  next to each generation.
- Add a documented `--deterministic` mode for deterministic algorithms where
  supported. Default behavior should remain performance-oriented but still
  reproducible enough for a fixed device/software stack.

### 4. Add checkpoints and exact resume

- At every validation interval, write `latest.pt` atomically with model state,
  optimizer state, scheduler state or recoverable step, AMP scaler, train
  iterator position, RNG states, histories, and all required config/provenance
  identifiers.
- When validation loss improves, atomically write `best-validation.pt`.
- Save `final.pt` after the requested maximum step without overwriting either
  checkpoint.
- Add `--resume <checkpoint>` and validate that resumed configuration,
  tokenizer ID, dataset digests, and architecture match the checkpoint before
  training continues. Permit an explicit escape hatch only for a documented
  non-reproducible fork.
- Define the initial best checkpoint as the first validation result, avoiding
  an uninitialized best-model path.

### 5. Add MPS support

- Extend device parsing to support `auto`, `cpu`, `cuda`, and `mps`.
- Resolve `auto` in priority order: CUDA, MPS, CPU.
- Enable CUDA AMP only on CUDA; MPS and CPU run without CUDA AMP unless a
  supported capability is deliberately added and covered by tests.
- Guard CUDA-only calls such as `empty_cache`, device-name reporting, and
  `GradScaler` initialization.
- Add device-selection tests that do not require a CUDA or MPS machine.

### 6. Verify and document

- Add unit tests for data hashing, IDs, checkpoint serialization, resume
  compatibility checks, and deterministic generation.
- Add a CPU smoke command using a small number of stories, a small model, and
  a few steps. It must produce all contract artifacts.
- Add a resume smoke test: train briefly, resume, and assert monotonic step
  progression plus matching provenance IDs.
- Run baseline inference against `xlarge-plus.pt` with
  `artifacts/tokenizers/tinystories_bpe_metaspace_5000_1000000.json` after
  verifying vocabulary compatibility. If it cannot reproduce the checkpoint's
  intended vocabulary, retain the baseline strictly as a historical artifact
  and retrain a new canonical baseline under the new contract.

## Acceptance Criteria

- A new CPU smoke run creates the complete artifact layout and passes tests.
- A seeded run produces identical sampled output on the same environment and
  invocation.
- A resumed run preserves step numbering, optimizer/scheduler state, and all
  provenance identities.
- The best-validation checkpoint, latest resumable checkpoint, and final model
  are distinct, present artifacts.
- `--device mps` is accepted on Apple Silicon and never invokes CUDA-only APIs.
- The baseline checkpoint is sampled using the explicitly verified recovered
  tokenizer; it is never silently paired with an incompatible tokenizer.

## Handoff Notes

This is suitable for a coding agent. It should make small, tested changes;
avoid retraining the 100,000-step baseline; and preserve the existing recovered
artifacts. Human review is required before changing the checkpoint format or
declaring the reconstructed tokenizer compatible with the historical model.
