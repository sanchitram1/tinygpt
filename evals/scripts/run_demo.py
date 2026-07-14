#!/usr/bin/env python3
"""Demo run of story-v0.1 against a hand-written mock backend.

This is NOT a real baseline run. `core/` does not yet expose a stable
inference API to plug into `eval_suite`'s `GenerateFn` Protocol (see
`../CORE_INTEGRATION_CONTRACT.md`) -- that work is happening concurrently in
a separate work stream. This script exists to produce a concrete, checked-in
example of the runner's JSONL output and human-review report end-to-end,
using the same canned mock generations the test suite uses, so reviewers can
see the pipeline before it is wired to a real checkpoint.

Once a real backend adapter satisfying `GenerateFn` exists, replace
`canned_generator_for(cases)` below with that adapter and a real
`ArtifactBundle`.
"""

from __future__ import annotations

import sys
from pathlib import Path

EVALS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVALS_ROOT))

from eval_suite.contracts import ArtifactBundle
from eval_suite.decoding_config import load_decoding_config
from eval_suite.report import write_report
from eval_suite.runner import (
    output_dir_for,
    run_suite,
    summarize_results,
    write_results_jsonl,
    write_summary_json,
)
from eval_suite.schema import load_dataset
from tests.mock_generation import canned_generator_for

DATASET_PATH = EVALS_ROOT / "datasets" / "story-v0.1.jsonl"
CONFIG_PATH = EVALS_ROOT / "configs" / "decoding-v0.1.json"
OUTPUT_ROOT = EVALS_ROOT / "results"


def main() -> None:
    decoding_config = load_decoding_config(CONFIG_PATH)
    artifact_bundle = ArtifactBundle(
        model_version="mock-demo",
        run_id="demo-run-001",
        tokenizer_id="mock-tokenizer",
    )
    cases = load_dataset(DATASET_PATH, suite_version="v0.1")
    generate_fn = canned_generator_for(cases)

    results = run_suite(
        DATASET_PATH,
        decoding_config=decoding_config,
        artifact_bundle=artifact_bundle,
        generate_fn=generate_fn,
    )
    summary = summarize_results(results)

    out_dir = output_dir_for(
        OUTPUT_ROOT,
        suite_version="v0.1",
        model_version=artifact_bundle.model_version,
        run_id=artifact_bundle.run_id,
    )
    write_results_jsonl(results, out_dir / "results.jsonl")
    write_summary_json(summary, out_dir / "summary.json")
    write_report(
        results,
        summary,
        out_dir / "report.md",
        suite_version="v0.1",
        model_version=artifact_bundle.model_version,
        run_id=artifact_bundle.run_id,
    )
    print(f"Wrote demo results to {out_dir}")


if __name__ == "__main__":
    main()
