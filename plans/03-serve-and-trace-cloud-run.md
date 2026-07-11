# Plan 3: Serve and Trace TinyGPT on Cloud Run

## Objective

Deliver a minimal public TinyGPT chat interface backed by a FastAPI service on
Google Cloud Run, and export every generation trace to an Arize Space/Project
using OpenTelemetry and OpenInference conventions.

The first deployment may serve the Plan 1 canonical baseline. A later Cloud Run
revision promotes a candidate from the future Sol-operated training phase only
after it passes the Plan 2 evaluation suite and a human decision gate.

## Architecture

```text
Browser
  -> FastAPI + static HTML/CSS/JS chat page (Cloud Run)
  -> TinyGPT inference service in the same process
  -> bundled model artifact or startup download from private Cloud Storage
  -> OTLP exporter -> Arize Space / Project
```

Do not fork a broad chat product. This service needs only a single conversation
view, a reset action, and a generation endpoint. A small in-repo FastAPI app
will be easier to connect to custom PyTorch inference and version metadata than
a general-purpose chat UI.

Cloud Run supports container deployments for FastAPI services, including
Dockerfile-based builds. Arize accepts OpenTelemetry spans enriched with
OpenInference attributes over OTLP, so manual instrumentation is appropriate
for TinyGPT's custom generation loop. See [Cloud Run's FastAPI
quickstart](https://cloud.google.com/run/docs/quickstarts/build-and-deploy/deploy-python-fastapi-service)
and [Arize tracing concepts](https://arize.com/docs/ax/observe/tracing/spans).

## Required IDs and Values

Provide these values when deployment work begins. Do not place secrets in the
repository or a `.env` file committed to git.

| Value | Why it is needed | Example |
|---|---|---|
| GCP project ID | Cloud Run and Artifact Registry target | `my-tinygpt-project` |
| Cloud Run region | Deployment and latency location | `us-west1` |
| Cloud Run service name | Stable service identity | `tinygpt-chat` |
| Artifact mode | Determines model delivery | `bundle-in-image` or `gcs-startup-download` |
| GCS bucket and object path | Required only for GCS artifact mode | `gs://my-bucket/models/<model-version>/` |
| Arize Space ID or name | Trace destination context | provided from Arize UI/CLI |
| Arize Project ID or name | Trace destination context | provided from Arize UI/CLI |
| Arize API key | Authenticates OTLP export | store as Cloud Run secret |
| Desired public-access policy | Public demo vs authenticated service | `public` or `IAP/private` |

The application should receive the Arize key and destination configuration from
Cloud Run environment variables/secrets. It must fail startup clearly when
tracing is enabled but required configuration is missing; a local development
mode may disable exporting explicitly.

## API and UI Scope

### Endpoints

- `GET /`: static single-page chat interface.
- `POST /api/chat`: accepts `message`, optional `session_id`, and optional
  decoding controls; returns generated story plus response metadata.
- `GET /healthz`: process health without loading/generating a story.
- `GET /readyz`: model/tokenizer readiness after artifact validation.

Conversation history is browser-local for the initial release. The backend
receives a bounded history or current prompt only, according to the model's
context budget. There is no database, account system, streaming, or durable
conversation storage in v1.

### Runtime safeguards

- Cap prompt and output tokens to the model context window and configured
  generation limit.
- Serialize model generation or set Cloud Run concurrency conservatively until
  memory/latency behavior is measured.
- Return a stable error response for invalid request bodies, overloaded model
  generation, and missing artifacts.
- Apply a simple request-size guard and rate-limit strategy before making a
  public endpoint broadly discoverable.

## Model Packaging

1. Read the selected checkpoint, tokenizer, and `manifest.json` from one Plan
   1 artifact bundle. Never load a checkpoint by a bare path without its
   associated tokenizer/provenance.
2. On application startup, verify model and tokenizer digests against the
   manifest, load the model once, move it to the supported serving device, and
   call `eval()`.
3. Start with `bundle-in-image` for the 185 MB baseline checkpoint. It makes
   the first deployment reproducible and avoids runtime bucket/IAM complexity.
4. Add GCS startup download only when image size or frequent model promotion
   justifies it. Use a private bucket and a dedicated Cloud Run service account.

## Trace Design

Each `POST /api/chat` creates one root request trace and a child generation
span. Use OpenInference-compatible input/output attributes where applicable,
then add the following project attributes to the generation span:

| Attribute | Source |
|---|---|
| `model_version` | Plan 1 artifact manifest |
| `run_id` | Plan 1 artifact manifest |
| `tokenizer_id` | Plan 1 artifact manifest |
| `decoding_parameters` | effective temperature, top-k, and max-new-tokens |
| `prompt_token_count` | encoded prompt length |
| `output_token_count` | tokens generated |
| `latency_ms` | generation wall time |
| `session_id` | client-provided or server-generated UUID |
| `synthetic` | request flag, default `false` |

Also record service revision, HTTP status, error type, model device, and a
request correlation ID. The `synthetic` attribute is mandatory for traffic from
load tests or scripted evaluation so it can be filtered from real-user traces.

Do not record credentials. Avoid adding personal identifiers to spans; a
random session UUID is sufficient for v1.

## Implementation Steps

1. Add a `service/` package with model-artifact loading, request/response
   schemas, inference adapter, FastAPI routes, static assets, and telemetry
   initialization.
2. Refactor generation to return text, input/output token counts, stop reason,
   and timing without changing the existing training API unexpectedly.
3. Add manual OpenTelemetry spans around request handling and TinyGPT
   generation. Validate locally with telemetry disabled, then with an Arize
   development destination.
4. Add a Dockerfile that installs the pinned project dependencies, copies one
   selected artifact bundle, exposes `$PORT`, and runs Uvicorn. Add `.dockerignore`
   so large unrelated data, artifacts, and local secrets are excluded.
5. Add local tests for request validation, artifact digest mismatch, trace
   attribute construction, and a mocked generation request.
6. Build and run the container locally. Verify `/healthz`, `/readyz`, an
   ordinary chat request, and a request with `synthetic=true`.
7. Provide a deploy guide containing build, Artifact Registry push, secret
   creation, and `gcloud run deploy` commands. The user executes those commands
   from their personal GCP environment unless they explicitly authorize access.
8. After deployment, send a tagged synthetic request and verify that its trace
   appears in the intended Arize Space/Project with all required attributes.

## Acceptance Criteria

- A local Docker build launches a usable chat page and serves a checkpoint with
  a verified matching tokenizer.
- The deployed Cloud Run revision has explicit model/run/tokenizer provenance.
- Every successful and failed generation exports a trace with all nine required
  attributes and no secrets.
- A synthetic probe is visible and filterable independently from normal chat
  traffic.
- A model promotion is a new immutable artifact bundle and Cloud Run revision,
  not an overwrite of a running model.

## Handoff Notes

This is suitable for a coding agent once the IDs above are available. The agent
can build the Dockerfile, FastAPI service, local tests, and deployment guide
without cloud credentials. It must not deploy, make the service public, or
create secrets without explicit user authorization.
