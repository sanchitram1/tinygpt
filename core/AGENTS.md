# Core

## Overview

`core/` contains reusable TinyGPT runtime primitives and immutable recovered
inputs. Training, evals, and the service must use this code rather than copy it.

## Standards

- Keep model loading, tokenizer loading, generation, configuration, and artifact
  identities independent of a training or serving environment.
- `core/data/` and `core/artifacts/` are read-only inputs. Never rewrite the
  recovered baseline or tokenizer artifacts.
- Validate checkpoint/tokenizer compatibility before inference.
- Make paths explicit and portable; do not infer Colab, RunPod, or Cloud Run
  from the machine environment.
- Preserve stable, typed interfaces. Avoid importing from `training/`, `evals/`,
  or `service/`.

## Verification

Add focused unit tests for any changed serialization, artifact validation, or
generation behavior. Run the relevant test command and `git diff --check`.
