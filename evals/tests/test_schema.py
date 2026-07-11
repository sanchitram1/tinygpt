from pathlib import Path

import pytest

from eval_suite.schema import (
    SchemaValidationError,
    load_dataset,
    validate_case,
    validate_dataset,
)

DATASET_PATH = Path(__file__).resolve().parents[1] / "datasets" / "story-v0.1.jsonl"


def test_story_v0_1_loads_and_has_exactly_ten_cases():
    cases = load_dataset(DATASET_PATH, suite_version="v0.1")
    assert len(cases) == 10


def test_story_v0_1_covers_each_category_exactly_twice():
    cases = load_dataset(DATASET_PATH, suite_version="v0.1")
    counts: dict[str, int] = {}
    for case in cases:
        counts[case.category] = counts.get(case.category, 0) + 1
    assert counts == {
        "basic-completion": 2,
        "prompt-adherence": 2,
        "tone": 2,
        "structural-constraints": 2,
        "robustness": 2,
    }


def test_story_v0_1_ids_are_unique():
    cases = load_dataset(DATASET_PATH, suite_version="v0.1")
    ids = [case.id for case in cases]
    assert len(ids) == len(set(ids))


def test_valid_case_passes():
    validate_case(
        {
            "id": "sample-case-001",
            "prompt": "Write a story.",
            "category": "basic-completion",
            "expected_attributes": {"child_safe": True},
        }
    )


def test_case_missing_child_safe_is_rejected():
    with pytest.raises(SchemaValidationError):
        validate_case(
            {
                "id": "sample-case-002",
                "prompt": "Write a story.",
                "category": "basic-completion",
                "expected_attributes": {},
            }
        )


def test_case_with_child_safe_false_is_rejected():
    with pytest.raises(SchemaValidationError):
        validate_case(
            {
                "id": "sample-case-003",
                "prompt": "Write a story.",
                "category": "basic-completion",
                "expected_attributes": {"child_safe": False},
            }
        )


def test_case_with_unknown_category_is_rejected():
    with pytest.raises(SchemaValidationError):
        validate_case(
            {
                "id": "sample-case-004",
                "prompt": "Write a story.",
                "category": "not-a-real-category",
                "expected_attributes": {"child_safe": True},
            }
        )


def test_duplicate_ids_are_rejected():
    record = {
        "id": "dup-001",
        "prompt": "Write a story.",
        "category": "basic-completion",
        "expected_attributes": {"child_safe": True},
    }
    with pytest.raises(SchemaValidationError, match="duplicate"):
        validate_dataset([record, dict(record)])


def test_wrong_case_count_for_versioned_suite_is_rejected():
    record = {
        "id": "only-case-001",
        "prompt": "Write a story.",
        "category": "basic-completion",
        "expected_attributes": {"child_safe": True},
    }
    with pytest.raises(SchemaValidationError, match="exactly 10"):
        validate_dataset([record], suite_version="v0.1")


def test_unversioned_dataset_skips_coverage_rules():
    record = {
        "id": "only-case-001",
        "prompt": "Write a story.",
        "category": "basic-completion",
        "expected_attributes": {"child_safe": True},
    }
    validate_dataset([record], suite_version=None)


def test_invalid_json_line_reports_line_number(tmp_path):
    bad_path = tmp_path / "story-v0.9.jsonl"
    bad_path.write_text('{"id": "a"}\nnot json\n', encoding="utf-8")
    with pytest.raises(SchemaValidationError, match="2: invalid JSON"):
        load_dataset(bad_path)
