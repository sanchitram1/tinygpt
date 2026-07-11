# Decoding configs

Each file locks the decoding settings and global suite seed for one suite
run, per `plans/02-build-initial-evaluation-suite.md`. Schema in
`eval_suite/decoding_config.py`.

- `decoding-v0.1.json` — used with `story-v0.1.jsonl`.

Fields:

| field | meaning |
|---|---|
| `version` | Config version, e.g. `"v0.1"`. Independent of the dataset's suite version. |
| `temperature` | Fixed sampling temperature for every case. |
| `top_k` | Fixed top-k for every case. |
| `max_new_tokens` | Fixed generation length cap for every case. |
| `suite_seed` | Global seed. Per-case seeds are derived deterministically from this plus each case id (`DecodingConfig.case_seed`), not hand-listed per case. |

Do not edit a released decoding config in place — add a new versioned file
if settings need to change, so past results stay reproducible against the
config that produced them.
