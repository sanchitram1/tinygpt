"""Evaluation runner: executes a dataset against an injected generation callable.

The runner never imports or calls a real model. It is handed an
``ArtifactBundle`` (identity only) and a ``GenerateFn`` (see
``eval_suite.contracts``) and treats both as opaque. This keeps the runner
usable today, against a mock or fixture backend, and later against whatever
``core/`` ends up exposing -- as long as that backend is adapted to the
``GenerateFn`` Protocol. See ``evals/CORE_INTEGRATION_CONTRACT.md``.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from eval_suite.case_evaluators import evaluate_case
from eval_suite.contracts import ArtifactBundle, GenerateFn, GenerationRequest
from eval_suite.decoding_config import DecodingConfig
from eval_suite.schema import DatasetCase, load_dataset

SUITE_VERSION_RE = re.compile(r"-v(\d+\.\d+)\.jsonl$")


class SuiteRunError(ValueError):
    pass


def suite_version_from_path(dataset_path: Path) -> str:
    """Derive the suite version (e.g. ``"v0.1"``) from a dataset filename.

    Enforces the naming convention (``story-v0.1.jsonl``) that keeps released
    suites immutable and unambiguous, per ``evals/AGENTS.md``.
    """
    match = SUITE_VERSION_RE.search(Path(dataset_path).name)
    if not match:
        raise SuiteRunError(
            f"dataset filename {dataset_path!r} does not match the required "
            "'<name>-v<major>.<minor>.jsonl' convention"
        )
    return f"v{match.group(1)}"


def run_case(
    case: DatasetCase,
    *,
    decoding_config: DecodingConfig,
    generate_fn: GenerateFn,
) -> dict[str, Any]:
    """Run one case through ``generate_fn`` and evaluate the result.

    Latency is measured here (wall-clock around the call) rather than
    reported by the backend, so a backend only needs to implement generation,
    not timing.
    """
    seed = decoding_config.case_seed(case.id)
    request = GenerationRequest(
        prompt=case.prompt,
        max_new_tokens=decoding_config.max_new_tokens,
        temperature=decoding_config.temperature,
        top_k=decoding_config.top_k,
        seed=seed,
    )
    start = time.perf_counter()
    result = generate_fn(request)
    latency_ms = (time.perf_counter() - start) * 1000

    evaluator_results = evaluate_case(case, result.text)

    return {
        "case_id": case.id,
        "category": case.category,
        "prompt": case.prompt,
        "generated_text": result.text,
        "stop_reason": result.stop_reason,
        "latency_ms": round(latency_ms, 3),
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "generation_seed": seed,
        "decoding": decoding_config.as_dict(),
        "evaluator_results": evaluator_results,
    }


def run_suite(
    dataset_path: Path,
    *,
    decoding_config: DecodingConfig,
    artifact_bundle: ArtifactBundle,
    generate_fn: GenerateFn,
) -> list[dict[str, Any]]:
    """Run every case in ``dataset_path`` and return one result record per case.

    Raises ``IncompatibleArtifactError`` if ``artifact_bundle`` is missing
    required identity fields, and ``SchemaValidationError``/``SuiteRunError``
    if the dataset is malformed or misnamed.
    """
    artifact_bundle.validate()
    suite_version = suite_version_from_path(dataset_path)
    cases = load_dataset(dataset_path, suite_version=suite_version)

    results = []
    for case in cases:
        record = run_case(case, decoding_config=decoding_config, generate_fn=generate_fn)
        record.update(
            {
                "suite_version": suite_version,
                "model_version": artifact_bundle.model_version,
                "run_id": artifact_bundle.run_id,
                "tokenizer_id": artifact_bundle.tokenizer_id,
            }
        )
        results.append(record)
    return results


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate pass/fail counts by evaluator and by category."""
    total = len(results)
    overall_pass_count = sum(
        1 for r in results if r["evaluator_results"].get("overall_pass")
    )

    evaluator_names = set()
    for record in results:
        evaluator_names.update(
            name for name in record["evaluator_results"] if name != "overall_pass"
        )

    by_evaluator: dict[str, dict[str, int]] = {}
    for name in sorted(evaluator_names):
        applicable = [
            r["evaluator_results"][name]
            for r in results
            if r["evaluator_results"][name].get("pass") is not None
        ]
        by_evaluator[name] = {
            "applicable_cases": len(applicable),
            "passed": sum(1 for entry in applicable if entry["pass"]),
        }

    by_category: dict[str, dict[str, int]] = {}
    for record in results:
        bucket = by_category.setdefault(record["category"], {"total": 0, "overall_pass": 0})
        bucket["total"] += 1
        if record["evaluator_results"].get("overall_pass"):
            bucket["overall_pass"] += 1

    return {
        "total_cases": total,
        "overall_pass_count": overall_pass_count,
        "by_evaluator": by_evaluator,
        "by_category": by_category,
    }


def write_results_jsonl(results: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in results:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def write_summary_json(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def output_dir_for(
    output_root: Path, *, suite_version: str, model_version: str, run_id: str
) -> Path:
    """The plan's ``<suite-version>/<model-version>/`` layout, plus a run_id
    level underneath so re-runs of the same model version never collide.
    """
    return Path(output_root) / suite_version / model_version / run_id
