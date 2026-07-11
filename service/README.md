# tinygpt-service

Minimal FastAPI chat wrapper for TinyGPT with a static single-conversation UI,
health/readiness endpoints, request limits, and Arize-bound trace-attribute
construction. Scope and acceptance criteria: [Plan 3](../plans/03-serve-and-trace-cloud-run.md).

## Status

Service scaffolding only. The real TinyGPT adapter is **not** integrated yet
because the Plan 1 core runtime API is still stabilizing; the app starts in
not-ready mode (`/readyz` and `/api/chat` return 503) until it lands. No
deployment, secrets, or Arize calls happen from this repo.

## Run locally

```sh
cd service
uv run uvicorn tinygpt_service.main:app --reload
```

Then open http://127.0.0.1:8000. `/healthz` works immediately; `/readyz` is 503
until the adapter is integrated.

## Tests

```sh
cd service
uv run pytest
```

Tests inject a fake `StoryGenerator`, so they need neither torch nor a model
artifact.

## Docker

Build from the repository root so the baseline artifact bundle can be baked in
(`core/artifacts/` is gitignored — it must exist in your checkout):

```sh
docker build -f service/Dockerfile -t tinygpt-chat .
docker run --rm -p 8080:8080 tinygpt-chat
```

## Configuration

All via environment variables; see `tinygpt_service/config.py` for the full
list and defaults. Tracing is off by default. Setting
`TINYGPT_TRACING_ENABLED=true` requires `ARIZE_SPACE_ID`, `ARIZE_API_KEY`
(as a Cloud Run secret, never in the repo), and `ARIZE_PROJECT_NAME`;
startup fails clearly if any are missing.

## Core interface needed for final integration

The service depends only on the `StoryGenerator` protocol in
`tinygpt_service/generator.py`. To wire up the real model, implement
`TinyGPTGenerator` (see its docstring) on top of the finalized core runtime:

- Load checkpoint + tokenizer + `manifest.json` from one artifact bundle
  directory, verifying digests against the manifest.
- Expose provenance: `model_version`, `run_id`, `tokenizer_id`, `device`,
  `context_length`.
- `generate(prompt, *, temperature, top_k, max_new_tokens)` returning text,
  prompt/output token counts, stop reason, and latency, enforcing the context
  budget, callable from a worker thread.

Then make `load_generator()` return it, and add torch/tokenizers via the
`[model]` extra.
