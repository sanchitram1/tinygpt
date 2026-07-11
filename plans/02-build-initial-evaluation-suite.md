# Plan 2: Build the Initial Evaluation Suite

## Objective

Create the small, versioned behavioral evaluation dataset and local harness
needed to judge the recovered baseline and any future training improvement.
This plan deliberately does **not** let GPT-5.6 Sol alter training. Sol-operated
experimentation is a later phase that depends on this suite being trusted
enough to catch behavioral regressions.

## Dependencies

- Plan 1 has verified the baseline checkpoint/tokenizer pair and exposed a
  stable local inference API from `core/`.
- The baseline's artifact manifest supplies `model_version`, `run_id`, and
  `tokenizer_id`.

## Scope

1. Author an exploratory, human-readable dataset of ten story-generation cases.
2. Build a local runner that executes the same cases against an immutable
   artifact bundle with fixed decoding settings and seeds.
3. Reuse and formalize the existing heuristic checks from
   `generation_quality.py`.
4. Generate a review report that makes human assessment fast and consistent.
5. Preserve a simple data and output shape that can later become an Arize
   dataset and experiment without a migration.

Arize dataset creation, hosted evaluators, and trace-to-dataset feedback are
out of scope for this first suite. Those belong in the subsequent Arize eval
loop plan.

## Dataset Design

Store the initial suite at `evals/datasets/story-v0.1.jsonl`. Each record must
include:

```json
{
  "id": "opening-gentle-rabbit-001",
  "prompt": "Write a gentle bedtime story about a rabbit named Nia. Start with 'Once upon a time'.",
  "category": "opening-and-tone",
  "expected_attributes": {
    "required_terms": ["Nia"],
    "required_opening": "Once upon a time",
    "tone": "gentle",
    "child_safe": true
  },
  "notes": "Tests literal opening and a named protagonist."
}
```

The ten cases should cover:

| Category | Count | Required behavior |
|---|---:|---|
| Basic story completion | 2 | Coherent continuation and clean ending |
| Prompt adherence | 2 | Required protagonist, setting, or object |
| Tone | 2 | Distinguish gentle/bedtime from silly/playful |
| Structural constraints | 2 | Required opening and ending or lesson |
| Robustness | 2 | Multiple named characters and an unusual but child-safe request |

The cases must be hand-authored, short, child-safe, and free from targets copied
from the training or validation corpus. A human approves version `v0.1` before
it is used for promotion decisions.

## Fixed Evaluation Protocol

- Run every case against one selected checkpoint/tokenizer artifact bundle.
- Use a fixed global suite seed and a recorded per-case generation seed.
- Lock `temperature`, `top_k`, and `max_new_tokens` in a versioned suite config.
- Run each candidate using the exact same cases and decoding protocol.
- Store outputs and metrics below
  `artifacts/evaluations/<suite-version>/<model-version>/`.
- Never modify a released suite in place; use `story-v0.2.jsonl` for additions
  or changed expectations.

Each output record must include case ID, generated text, stop reason, latency,
prompt/output token counts, model version, run ID, tokenizer ID, decoding
parameters, generation seed, and evaluator results.

## Initial Evaluators

### Deterministic code evaluators

- Required literal terms and requested opening.
- Clean ending.
- Bigram and trigram repetition ratios.
- Unexpected-word count.
- Entity/confusion metrics from `generation_quality.py`.
- Output length range.

These are diagnostic signals, not a claim that they fully measure story quality.
Each evaluator reports raw values plus a clear pass/fail threshold where one is
appropriate.

### Human review rubric

Generate a compact Markdown or HTML report showing prompt, expected attributes,
output, code metrics, and four 1-5 ratings:

1. Prompt adherence.
2. Coherence and narrative flow.
3. Child appropriateness.
4. Overall story quality.

The reviewer can add a short issue tag such as `repetition`, `lost-character`,
`unsafe`, `ignored-constraint`, or `unfinished`.

No LLM judge is part of `v0.1`. First calibrate the suite with direct human
review; an LLM judge can later be compared against the accumulated labels.

## Implementation Steps

1. Create the `evals/` package, dataset schema validation, `story-v0.1.jsonl`,
   versioned decoding config, and a short dataset README.
2. Extract import-safe quality functions from `generation_quality.py` without
   altering its existing CLI behavior unexpectedly.
3. Implement an evaluation runner that accepts an artifact bundle, suite path,
   and output root. It must reject incompatible checkpoint/tokenizer manifests.
4. Write JSONL results and an aggregate summary; generate the human review
   report from the same results rather than a separate manual export.
5. Add tests for schema validation, deterministic run metadata, metric outputs,
   and report construction with mocked generation.
6. Run `story-v0.1` on the recovered baseline and have a human score the report.
7. Document observed baseline failures and use them to define the first future
   Sol training hypothesis. Do not change the suite to hide those failures.

## Acceptance Criteria

- `story-v0.1` has exactly ten approved cases with the stated coverage.
- The baseline run produces complete, reproducible result records and report.
- Re-running the same model/suite/configuration on the same environment yields
  the same outputs and evaluator results.
- A human can score all ten cases without opening raw JSON or training logs.
- The dataset and result schema are directly mappable to an Arize dataset and
  experiment run in the next phase.

## Future Dependency: Sol-Operated Experiments

The later Sol phase starts only after the human accepts the baseline report.
It will receive a bounded GPU budget, immutable suite version, one-hypothesis
experiment policy, and human promotion gate. It may propose changes, but it may
not rewrite the suite or promote a model to serving.
