# Core integration contract

`evals/` is being built while `core/`'s inference API is still being
stabilized in a separate work stream (Plan 1). Rather than guess at that
API's shape, the runner in `eval_suite/runner.py` depends only on the small
interface defined in `eval_suite/contracts.py`. This document is the
handoff spec: whatever `core/` ends up exposing, an adapter satisfying this
contract is all `evals/` needs to run against a real checkpoint.

## What the runner needs, and nothing more

### 1. A `generate_fn` callable

```python
def generate_fn(request: GenerationRequest) -> GenerationResult: ...
```

**Input — `GenerationRequest`** (`eval_suite/contracts.py`):

| field | type | meaning |
|---|---|---|
| `prompt` | `str` | The case's prompt text, verbatim from the dataset. |
| `max_new_tokens` | `int` | From the versioned decoding config. |
| `temperature` | `float` | From the versioned decoding config. |
| `top_k` | `int` | From the versioned decoding config. |
| `seed` | `int` | Per-case seed derived from the suite seed + case id (`DecodingConfig.case_seed`). The backend must use this to make generation for a given case reproducible. |

**Output — `GenerationResult`**:

| field | type | meaning |
|---|---|---|
| `text` | `str` | The generated continuation (or full completion, per whatever convention `core/` uses — the runner treats it as opaque text). |
| `stop_reason` | `str` | One of `"eos"`, `"max_tokens"`, `"stop_sequence"`, `"other"` (`eval_suite.contracts.STOP_REASONS`). Map whatever `core/`'s generation loop reports onto this small vocabulary. |
| `prompt_tokens` | `int` | Token count of the encoded prompt. |
| `completion_tokens` | `int` | Token count of the generated text. |

That's the entire generation surface. Deliberately **not** part of this
contract:

- **Latency.** The runner measures wall-clock time around the call to
  `generate_fn` itself (`eval_suite/runner.py:run_case`), so a backend adapter
  does not need to instrument timing.
- **Model/tokenizer identity.** Not repeated per call. See below.
- **Batching.** The runner calls `generate_fn` once per case, synchronously.
  If `core/` only exposes batched generation, the adapter is responsible for
  wrapping it into this one-call-per-case shape.

### 2. An `ArtifactBundle`

```python
ArtifactBundle(model_version: str, run_id: str, tokenizer_id: str)
```

Supplied once per run (not per generation call), and validated up front —
`run_suite` raises `IncompatibleArtifactError` if any of the three fields is
empty. This is intentionally a flat, backend-agnostic record rather than a
reference to `core/artifacts/baseline/manifest.json`'s current shape, since
that manifest has no `model_version` or `tokenizer_id` field today (it has
`run_id` and a `models` list of architecture configs — see that file).
Whoever wires up a real backend maps their manifest onto this bundle, e.g.:

```python
manifest = json.loads(Path("core/artifacts/baseline/manifest.json").read_text())
bundle = ArtifactBundle(
    model_version=manifest["models"][-1]["name"],   # e.g. "xlarge-plus"
    run_id=manifest["run_id"],
    tokenizer_id=derive_tokenizer_id(manifest),      # not yet defined by core/
)
```

`derive_tokenizer_id` is a placeholder: `core/` does not yet publish a
tokenizer identifier distinct from a tokenizer file path. Until it does, an
adapter can use the tokenizer artifact's filename stem (e.g.
`tinystories_bpe_metaspace_5000_1000000`) as a stable-enough id.

## What `evals/` does today instead

`evals/tests/mock_generation.py` implements this Protocol with no model at
all — canned, hand-written text keyed by prompt, or a pure function of the
request for determinism tests. `run_suite` cannot tell the difference between
that and a real backend; that is the point of depending on the Protocol
instead of a concrete `core/` API.

## Checklist for whoever wires up the real backend

1. Write an adapter function/class matching `GenerateFn`'s signature that
   calls into whatever `core/` exposes (model load, tokenizer encode/decode,
   sampling loop).
2. Build one `ArtifactBundle` from the manifest of the checkpoint/tokenizer
   pair being evaluated.
3. Call `eval_suite.runner.run_suite(dataset_path, decoding_config=...,
   artifact_bundle=..., generate_fn=adapter)`.
4. Write results with `write_results_jsonl` / `write_summary_json`, and a
   report with `eval_suite.report.write_report`.

No other change to `eval_suite/` should be required. If a real backend needs
something this contract doesn't provide (e.g. it cannot report exact token
counts), that is a signal to extend `GenerationResult` deliberately, not to
special-case the runner.
