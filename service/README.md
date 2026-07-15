# tinygpt-service

Minimal FastAPI chat wrapper for TinyGPT with a static single-conversation UI,
health/readiness endpoints, request limits, and Arize-bound trace-attribute
construction. Scope and acceptance criteria: [Plan 3](../plans/03-serve-and-trace-cloud-run.md).

## Status

The service loads one verified model bundle during application startup. Missing
or invalid bundles fail startup clearly; no deployment, secrets, or Arize calls
happen from this repo.

## Run locally

```sh
cd service
PYTHONPATH=.. uv run uvicorn tinygpt_service.main:app --reload
```

Set `TINYGPT_BUNDLE_DIR` to a bundle directory before starting the service,
then open http://127.0.0.1:8000. `/readyz` reports the loaded model provenance.

## Tests

```sh
cd service
uv run pytest
```

Tests inject a fake `StoryGenerator`, so they need neither torch nor a model
artifact.

## Build a bundle

The bundle command copies a checkpoint and tokenizer into a new directory and
writes a manifest containing model metadata and SHA-256 digests. It refuses to
overwrite an existing bundle or modify the source artifacts:

```sh
cd service
PYTHONPATH=.. uv run --extra model tinygpt-build-bundle \
  --checkpoint ../artifacts/runs/hw3-submission/models/5_xlarge-plus.pt \
  --tokenizer ../artifacts/shared/tokenizers/tinystories_bpe_metaspace_5000_1000000.json \
  --output ../model-bundles/xlarge-plus
```

Run locally with `TINYGPT_BUNDLE_DIR=../model-bundles/xlarge-plus` and choose a
device with `TINYGPT_DEVICE=auto|cpu|mps|cuda`.

## Docker

Build from the repository root after creating `model-bundles/xlarge-plus`:

```sh
docker build -f service/Dockerfile -t tinygpt-chat .
docker run --rm -p 8080:8080 tinygpt-chat
```

Use `--build-arg BUNDLE_SOURCE=/path/relative/to/repository` to bake a
different immutable bundle into the image.

## Configuration

All via environment variables; see `tinygpt_service/config.py` for the full
list and defaults. Tracing is off by default. Setting
`TINYGPT_TRACING_ENABLED=true` requires `ARIZE_SPACE_ID`, `ARIZE_API_KEY`
(as a Cloud Run secret, never in the repo), and `ARIZE_PROJECT_NAME`;
startup fails clearly if any are missing.

The browser uses `POST /api/chat/stream`, which emits Server-Sent Events with
text deltas followed by a final metadata event. `POST /api/chat` remains
available for complete JSON responses.
