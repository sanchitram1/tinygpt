from pathlib import Path

from eval_suite.contracts import ArtifactBundle
from eval_suite.decoding_config import load_decoding_config
from eval_suite.report import ISSUE_TAGS, RATING_DIMENSIONS, build_report, write_report
from eval_suite.runner import run_suite, summarize_results
from eval_suite.schema import load_dataset

from tests.mock_generation import canned_generator_for

DATASET_PATH = Path(__file__).resolve().parents[1] / "datasets" / "story-v0.1.jsonl"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "decoding-v0.1.json"


def _sample_results():
    decoding_config = load_decoding_config(CONFIG_PATH)
    artifact_bundle = ArtifactBundle(
        model_version="xlarge-plus-mock", run_id="test-run-001", tokenizer_id="tok-5000"
    )
    cases = load_dataset(DATASET_PATH, suite_version="v0.1")
    generate_fn = canned_generator_for(cases)
    results = run_suite(
        DATASET_PATH,
        decoding_config=decoding_config,
        artifact_bundle=artifact_bundle,
        generate_fn=generate_fn,
    )
    return results, summarize_results(results)


def test_report_includes_every_case_id():
    results, summary = _sample_results()
    report = build_report(
        results, summary, suite_version="v0.1", model_version="xlarge-plus-mock", run_id="test-run-001"
    )
    for record in results:
        assert record["case_id"] in report


def test_report_includes_prompt_and_generated_text():
    results, summary = _sample_results()
    report = build_report(
        results, summary, suite_version="v0.1", model_version="xlarge-plus-mock", run_id="test-run-001"
    )
    assert results[0]["prompt"] in report
    assert results[0]["generated_text"] in report


def test_report_includes_all_rating_dimensions_per_case():
    results, summary = _sample_results()
    report = build_report(
        results, summary, suite_version="v0.1", model_version="xlarge-plus-mock", run_id="test-run-001"
    )
    for dimension in RATING_DIMENSIONS:
        assert report.count(dimension) == len(results)


def test_report_lists_issue_tags():
    results, summary = _sample_results()
    report = build_report(
        results, summary, suite_version="v0.1", model_version="xlarge-plus-mock", run_id="test-run-001"
    )
    for tag in ISSUE_TAGS:
        assert tag in report


def test_report_does_not_require_opening_raw_json(tmp_path):
    results, summary = _sample_results()
    report = build_report(
        results, summary, suite_version="v0.1", model_version="xlarge-plus-mock", run_id="test-run-001"
    )
    # A human reviewer must be able to see prompt, expected attributes worth
    # of context (via case metrics), and output as prose/tables, not raw JSON
    # blobs.
    assert "{'text':" not in report
    assert '"generated_text"' not in report


def test_write_report_creates_file(tmp_path):
    results, summary = _sample_results()
    out_path = tmp_path / "nested" / "report.md"
    write_report(
        results,
        summary,
        out_path,
        suite_version="v0.1",
        model_version="xlarge-plus-mock",
        run_id="test-run-001",
    )
    assert out_path.exists()
    assert "# Story eval report" in out_path.read_text(encoding="utf-8")
