# Service

## Overview

This directory owns the minimal FastAPI chat wrapper, static UI, container
configuration, and Arize tracing. Read [Plan 3](../plans/03-serve-and-trace-cloud-run.md).

## Standards

- Load only a verified checkpoint/tokenizer artifact bundle from `../core/`.
- Keep v1 simple: no durable chat history, account system, or database.
- Enforce prompt/output limits and return stable errors for invalid requests or
  unavailable model artifacts.
- Each generation trace must include model version, run ID, tokenizer ID,
  decoding parameters, token counts, latency, session ID, and `synthetic`.
- Never commit credentials or deploy, create secrets, or enable public access
  without explicit approval.

## Verification

Test health/readiness routes, request validation, artifact mismatch handling,
and trace-attribute construction. Build the Docker image locally before any
Cloud Run handoff.
