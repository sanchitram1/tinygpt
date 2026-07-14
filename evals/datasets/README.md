# Story evaluation datasets

## `story-v0.1.jsonl`

Ten hand-authored, child-safe story-generation cases, two per category:

| Category | Count | Required behavior |
|---|---:|---|
| `basic-completion` | 2 | Coherent continuation and clean ending |
| `prompt-adherence` | 2 | Required protagonist, setting, or object |
| `tone` | 2 | Distinguish gentle/bedtime from silly/playful |
| `structural-constraints` | 2 | Required opening and ending or lesson |
| `robustness` | 2 | Multiple named characters and an unusual but child-safe request |

Every case is original — none of the prompts or expected content are copied
from the TinyStories training or validation corpus.

## Record format

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

`expected_attributes.child_safe` is required and must be `true`; every other
`expected_attributes` key is optional and only present when that case tests
it. See `eval_suite/schema.py` for the enforced JSON Schema and
`eval_suite/case_evaluators.py` for how each key is checked against generated
text.

## Status

`story-v0.1` is a draft authored for this implementation pass. Per
`evals/AGENTS.md`, a human must review and approve it before it is used for
any promotion decision — treat it as pending human sign-off, not yet a
released gate.

## Versioning

Released suites are immutable. To change a case's prompt or expectations,
create `story-v0.2.jsonl` (or later) rather than editing this file in place.
`eval_suite.runner.suite_version_from_path` derives the suite version from
the filename, so it must follow the `<name>-v<major>.<minor>.jsonl`
convention.
