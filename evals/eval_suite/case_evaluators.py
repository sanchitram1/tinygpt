"""Deterministic, code-based evaluators for a single case's generated text.

These are diagnostic signals, not a substitute for human review (see
``evals/AGENTS.md``). Each evaluator reports a raw value and, where a
threshold is meaningful, a pass/fail boolean. Evaluators tied to an
``expected_attributes`` key that a case does not set report
``"applicable": False`` and ``"pass": None`` rather than being omitted, so the
report can show every evaluator for every case.
"""

from __future__ import annotations

from typing import Any

from eval_suite import quality_metrics
from eval_suite.schema import DatasetCase

MAX_BIGRAM_REPETITION = 0.5
MAX_TRIGRAM_REPETITION = 0.3
MAX_WEIRD_WORDS = 3
MAX_ADJACENT_NAME_SWITCHES = 1
MIN_WORDS = 15
MAX_WORDS = 300

LESSON_PHRASES = (
    "learned",
    "lesson",
    "from that day",
    "never again",
    "realized that",
    "moral",
)


def _required_terms(case: DatasetCase, text: str) -> dict[str, Any]:
    required = case.expected_attributes.get("required_terms")
    if not required:
        return {"applicable": False, "pass": None, "required": [], "missing": []}
    lowered = text.lower()
    missing = [term for term in required if term.lower() not in lowered]
    return {
        "applicable": True,
        "required": list(required),
        "missing": missing,
        "pass": not missing,
    }


def _required_opening(case: DatasetCase, text: str) -> dict[str, Any]:
    opening = case.expected_attributes.get("required_opening")
    if not opening:
        return {"applicable": False, "pass": None, "expected": None}
    actual_start = text.strip()[: len(opening)]
    passed = actual_start.lower() == opening.lower()
    return {"applicable": True, "expected": opening, "pass": passed}


def _required_ending(case: DatasetCase, text: str) -> dict[str, Any]:
    ending = case.expected_attributes.get("required_ending")
    if not ending:
        return {"applicable": False, "pass": None, "expected": None}
    actual_end = text.rstrip()[-len(ending) :]
    passed = actual_end.lower() == ending.lower()
    return {"applicable": True, "expected": ending, "pass": passed}


def _required_lesson(case: DatasetCase, text: str) -> dict[str, Any]:
    if not case.expected_attributes.get("required_lesson"):
        return {"applicable": False, "pass": None}
    lowered = text.lower()
    matched = [phrase for phrase in LESSON_PHRASES if phrase in lowered]
    return {"applicable": True, "matched_phrases": matched, "pass": bool(matched)}


def _clean_ending(text: str) -> dict[str, Any]:
    return {"applicable": True, "pass": quality_metrics.ends_cleanly(text)}


def _repetition(text: str) -> dict[str, Any]:
    bigram = quality_metrics.bigram_repetition_ratio(text)
    trigram = quality_metrics.trigram_repetition_ratio(text)
    return {
        "applicable": True,
        "bigram_repetition_ratio": bigram,
        "trigram_repetition_ratio": trigram,
        "pass": bigram <= MAX_BIGRAM_REPETITION and trigram <= MAX_TRIGRAM_REPETITION,
    }


def _weird_words(text: str) -> dict[str, Any]:
    count = quality_metrics.weird_word_count(text)
    return {"applicable": True, "count": count, "pass": count <= MAX_WEIRD_WORDS}


def _entity_confusion(case: DatasetCase, text: str) -> dict[str, Any]:
    metrics = quality_metrics.entity_metrics(text)
    passed = metrics["adjacent_name_switches"] <= MAX_ADJACENT_NAME_SWITCHES
    if case.expected_attributes.get("multiple_characters"):
        required = case.expected_attributes.get("required_terms") or []
        passed = passed and metrics["unique_names"] >= max(2, len(required))
    return {"applicable": True, "pass": passed, **metrics}


def _output_length(text: str) -> dict[str, Any]:
    word_count = len(quality_metrics.words(text))
    return {
        "applicable": True,
        "word_count": word_count,
        "min_words": MIN_WORDS,
        "max_words": MAX_WORDS,
        "pass": MIN_WORDS <= word_count <= MAX_WORDS,
    }


def evaluate_case(case: DatasetCase, generated_text: str) -> dict[str, Any]:
    """Run every deterministic evaluator against one case's generated text."""
    results = {
        "required_terms": _required_terms(case, generated_text),
        "required_opening": _required_opening(case, generated_text),
        "required_ending": _required_ending(case, generated_text),
        "required_lesson": _required_lesson(case, generated_text),
        "clean_ending": _clean_ending(generated_text),
        "repetition": _repetition(generated_text),
        "weird_words": _weird_words(generated_text),
        "entity_confusion": _entity_confusion(case, generated_text),
        "output_length": _output_length(generated_text),
    }
    applicable_pass_values = [
        result["pass"] for result in results.values() if result.get("pass") is not None
    ]
    results["overall_pass"] = all(applicable_pass_values) if applicable_pass_values else None
    return results
