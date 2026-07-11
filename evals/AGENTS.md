# Evals

## Overview

This directory defines versioned story-quality cases, deterministic local
evaluation, and human-review reports. Read [Plan 2](../plans/02-build-initial-evaluation-suite.md).

## Standards

- Released datasets are immutable. Create a new suite version for changed cases
  or expectations.
- Keep evaluation inputs independent from training and validation examples.
- Record artifact identity, decoding parameters, seeds, token counts, latency,
  generated output, and evaluator results for every case.
- Treat heuristic metrics as diagnostics, not a substitute for human review.
- Do not change the suite to conceal a model regression.

## Verification

Validate dataset schema and run deterministic tests with mocked generation.
Review the generated report before treating a suite as a promotion gate.
