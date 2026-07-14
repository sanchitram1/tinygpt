# evals

Local behavioral evaluation suite for tinygpt story generation. Read
`AGENTS.md` and `../plans/02-build-initial-evaluation-suite.md` first.

## Layout

- `datasets/story-v0.1.jsonl` — ten hand-authored, child-safe cases. See
  `datasets/README.md`.
- `configs/decoding-v0.1.json` — versioned decoding settings + suite seed.
  See `configs/README.md`.
- `eval_suite/` — the runner package:
  - `schema.py` — dataset JSON Schema + loader/validator.
  - `decoding_config.py` — decoding config schema + per-case seed derivation.
  - `contracts.py` — the `GenerateFn`/`GenerationRequest`/`GenerationResult`/
    `ArtifactBundle` Protocol the runner depends on instead of a concrete
    `core/` inference API. See `CORE_INTEGRATION_CONTRACT.md`.
  - `quality_metrics.py` — text metrics extracted from `generation_quality.py`
    (repetition ratios, weird-word count, entity/pronoun confusion, clean
    ending).
  - `case_evaluators.py` — per-case deterministic pass/fail evaluators built
    on top of `quality_metrics` and each case's `expected_attributes`.
  - `runner.py` — `run_suite(...)`: runs every case through an injected
    `generate_fn`, evaluates it, and returns result records.
  - `report.py` — builds the human-review Markdown report from result
    records.
- `generation_quality.py` — original CLI for summarizing a batch of raw
  generations; unchanged behavior, now importing its metric functions from
  `eval_suite.quality_metrics`.
- `scripts/run_demo.py` — runs `story-v0.1` against a hand-written mock
  backend (not a real model) and writes results/summary/report under
  `results/`. This is the only "generation" that happens in this package
  today; see `CORE_INTEGRATION_CONTRACT.md` for why.
- `results/` — output of `scripts/run_demo.py`. `run_suite`'s callers choose
  the output root; nothing here is hardcoded into `eval_suite`.
- `tests/` — schema, decoding config, evaluator, runner, and report tests,
  all against mocked generation (`tests/mock_generation.py`).

## Running things

```bash
cd evals
uv sync
uv run pytest -q
uv run ruff check .
uv run python scripts/run_demo.py
```

## Status

`story-v0.1` is a draft pending human review and approval (see
`datasets/README.md`). The demo run in `results/` uses a mock backend, not
the recovered baseline checkpoint — `core/` does not yet expose a stable
inference API for `eval_suite` to call. Once it does, see
`CORE_INTEGRATION_CONTRACT.md` for the minimal adapter needed to run this
suite against a real checkpoint.
