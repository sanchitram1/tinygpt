"""Build a human-review Markdown report from evaluation result records.

The report is generated directly from the same JSONL result records written
by the runner (not a separate manual export), per
``plans/02-build-initial-evaluation-suite.md``. It shows prompt, expected
attributes, output, deterministic code metrics, and blank fields for the four
required 1-5 human ratings plus an optional issue tag.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

RATING_DIMENSIONS = (
    "Prompt adherence",
    "Coherence and narrative flow",
    "Child appropriateness",
    "Overall story quality",
)

ISSUE_TAGS = ("repetition", "lost-character", "unsafe", "ignored-constraint", "unfinished")


def _format_pass(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return "PASS" if value else "FAIL"


def _case_section(record: dict[str, Any]) -> str:
    evaluators = record["evaluator_results"]
    lines = [
        f"## {record['case_id']} ({record['category']})",
        "",
        f"**Overall deterministic result:** {_format_pass(evaluators.get('overall_pass'))}",
        "",
        "**Prompt:**",
        "",
        f"> {record['prompt']}",
        "",
        "**Generated output:**",
        "",
        f"> {record['generated_text']}",
        "",
        "**Run metadata:**",
        "",
        "| field | value |",
        "|---|---|",
        f"| stop_reason | {record['stop_reason']} |",
        f"| latency_ms | {record['latency_ms']} |",
        f"| prompt_tokens | {record['prompt_tokens']} |",
        f"| completion_tokens | {record['completion_tokens']} |",
        f"| generation_seed | {record['generation_seed']} |",
        f"| model_version | {record['model_version']} |",
        f"| run_id | {record['run_id']} |",
        f"| tokenizer_id | {record['tokenizer_id']} |",
        "",
        "**Code metrics:**",
        "",
        "| evaluator | result |",
        "|---|---|",
    ]
    for name, result in evaluators.items():
        if name == "overall_pass":
            continue
        detail = {k: v for k, v in result.items() if k not in ("applicable", "pass")}
        lines.append(f"| {name} | {_format_pass(result.get('pass'))} — `{detail}` |")

    lines += [
        "",
        "**Human ratings (1-5):**",
        "",
        "| dimension | rating | reviewer notes |",
        "|---|---|---|",
    ]
    for dimension in RATING_DIMENSIONS:
        lines.append(f"| {dimension} | _ | |")

    lines += [
        "",
        f"**Issue tag** (optional, one of {', '.join(ISSUE_TAGS)}): _",
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


def build_report(
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    suite_version: str,
    model_version: str,
    run_id: str,
) -> str:
    header = [
        f"# Story eval report: {suite_version} / {model_version}",
        "",
        f"- run_id: {run_id}",
        f"- total cases: {summary['total_cases']}",
        f"- cases passing all applicable deterministic evaluators: {summary['overall_pass_count']}",
        "",
        "## Deterministic summary by category",
        "",
        "| category | total | overall_pass |",
        "|---|---:|---:|",
    ]
    for category, bucket in summary["by_category"].items():
        header.append(f"| {category} | {bucket['total']} | {bucket['overall_pass']} |")

    header += [
        "",
        "## Deterministic summary by evaluator",
        "",
        "| evaluator | applicable cases | passed |",
        "|---|---:|---:|",
    ]
    for name, bucket in summary["by_evaluator"].items():
        header.append(f"| {name} | {bucket['applicable_cases']} | {bucket['passed']} |")

    header += ["", "## Cases", ""]

    sections = [_case_section(record) for record in results]
    return "\n".join(header) + "\n\n" + "\n".join(sections)


def write_report(
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    path: Path,
    *,
    suite_version: str,
    model_version: str,
    run_id: str,
) -> None:
    report = build_report(
        results,
        summary,
        suite_version=suite_version,
        model_version=model_version,
        run_id=run_id,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
