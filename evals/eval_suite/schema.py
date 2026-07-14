"""Dataset schema and validation for versioned story-eval suites.

A suite is a JSONL file where each line is one case. See
``evals/datasets/README.md`` for the case format and
``plans/02-build-initial-evaluation-suite.md`` for the design rationale.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema

CATEGORIES = (
    "basic-completion",
    "prompt-adherence",
    "tone",
    "structural-constraints",
    "robustness",
)

CASE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "story-eval-case",
    "type": "object",
    "required": ["id", "prompt", "category", "expected_attributes"],
    "additionalProperties": False,
    "properties": {
        "id": {
            "type": "string",
            "pattern": r"^[a-z0-9]+(-[a-z0-9]+)*$",
        },
        "prompt": {"type": "string", "minLength": 1},
        "category": {"type": "string", "enum": list(CATEGORIES)},
        "expected_attributes": {
            "type": "object",
            "required": ["child_safe"],
            "additionalProperties": False,
            "properties": {
                "required_terms": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 1,
                },
                "required_opening": {"type": "string", "minLength": 1},
                "required_ending": {"type": "string", "minLength": 1},
                "required_lesson": {"type": "boolean"},
                "tone": {"type": "string", "enum": ["gentle", "silly"]},
                "multiple_characters": {"type": "boolean"},
                "unusual_request": {"type": "boolean"},
                "child_safe": {"const": True},
            },
        },
        "notes": {"type": "string"},
    },
}

# Coverage rules for a specific released suite version, per the plan's table
# of required category counts. Keyed by suite version so future suites
# (story-v0.2, ...) can define their own coverage without touching this one.
SUITE_COVERAGE_RULES: dict[str, dict[str, Any]] = {
    "v0.1": {
        "total_cases": 10,
        "per_category": {category: 2 for category in CATEGORIES},
    },
}


class SchemaValidationError(ValueError):
    """Raised when a case or dataset fails validation."""


def validate_case(record: dict[str, Any]) -> None:
    """Validate a single case record against ``CASE_SCHEMA``.

    Raises ``SchemaValidationError`` with the offending record's ``id`` (if
    present) and the underlying validation error.
    """
    try:
        jsonschema.validate(instance=record, schema=CASE_SCHEMA)
    except jsonschema.ValidationError as exc:
        case_id = record.get("id", "<missing id>")
        raise SchemaValidationError(f"case {case_id!r} failed validation: {exc.message}") from exc


def validate_dataset(
    records: list[dict[str, Any]], *, suite_version: str | None = None
) -> None:
    """Validate every case plus dataset-level invariants (unique ids, coverage).

    If ``suite_version`` matches a key in ``SUITE_COVERAGE_RULES``, also
    enforces the exact case count and per-category count for that release.
    """
    for record in records:
        validate_case(record)

    ids = [record["id"] for record in records]
    duplicates = [item for item, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise SchemaValidationError(f"duplicate case ids: {sorted(duplicates)}")

    if suite_version is None or suite_version not in SUITE_COVERAGE_RULES:
        return

    rules = SUITE_COVERAGE_RULES[suite_version]
    if len(records) != rules["total_cases"]:
        raise SchemaValidationError(
            f"suite {suite_version} must have exactly {rules['total_cases']} cases, "
            f"found {len(records)}"
        )

    category_counts = Counter(record["category"] for record in records)
    for category, expected_count in rules["per_category"].items():
        actual_count = category_counts.get(category, 0)
        if actual_count != expected_count:
            raise SchemaValidationError(
                f"suite {suite_version} category {category!r} must have "
                f"{expected_count} cases, found {actual_count}"
            )


@dataclass(frozen=True)
class DatasetCase:
    id: str
    prompt: str
    category: str
    expected_attributes: dict[str, Any]
    notes: str = ""

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "DatasetCase":
        return cls(
            id=record["id"],
            prompt=record["prompt"],
            category=record["category"],
            expected_attributes=record["expected_attributes"],
            notes=record.get("notes", ""),
        )


def load_dataset(path: Path, *, suite_version: str | None = None) -> list[DatasetCase]:
    """Load a JSONL dataset file, validate it, and return ``DatasetCase`` objects."""
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SchemaValidationError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc

    validate_dataset(records, suite_version=suite_version)
    return [DatasetCase.from_record(record) for record in records]
