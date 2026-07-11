import json
from pathlib import Path

import pytest

from eval_suite.contracts import ArtifactBundle, IncompatibleArtifactError
from eval_suite.decoding_config import DecodingConfig, load_decoding_config
from eval_suite.runner import (
    SuiteRunError,
    output_dir_for,
    run_suite,
    suite_version_from_path,
    summarize_results,
    write_results_jsonl,
    write_summary_json,
)
from eval_suite.schema import load_dataset

from tests.mock_generation import (
    BrokenGenerator,
    DeterministicSeedEchoGenerator,
    canned_generator_for,
)

DATASET_PATH = Path(__file__).resolve().parents[1] / "datasets" / "story-v0.1.jsonl"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "decoding-v0.1.json"


@pytest.fixture
def decoding_config() -> DecodingConfig:
    return load_decoding_config(CONFIG_PATH)


@pytest.fixture
def artifact_bundle() -> ArtifactBundle:
    return ArtifactBundle(
        model_version="xlarge-plus-mock",
        run_id="test-run-001",
        tokenizer_id="tinystories_bpe_metaspace_5000_1000000",
    )


def test_suite_version_from_path():
    assert suite_version_from_path(DATASET_PATH) == "v0.1"


def test_suite_version_from_path_rejects_bad_filename(tmp_path):
    bad_path = tmp_path / "story.jsonl"
    with pytest.raises(SuiteRunError):
        suite_version_from_path(bad_path)


def test_run_suite_produces_one_record_per_case(decoding_config, artifact_bundle):
    cases = load_dataset(DATASET_PATH, suite_version="v0.1")
    generate_fn = canned_generator_for(cases)
    results = run_suite(
        DATASET_PATH,
        decoding_config=decoding_config,
        artifact_bundle=artifact_bundle,
        generate_fn=generate_fn,
    )
    assert len(results) == 10
    assert {r["case_id"] for r in results} == {c.id for c in cases}


def test_run_suite_records_required_output_fields(decoding_config, artifact_bundle):
    cases = load_dataset(DATASET_PATH, suite_version="v0.1")
    generate_fn = canned_generator_for(cases)
    results = run_suite(
        DATASET_PATH,
        decoding_config=decoding_config,
        artifact_bundle=artifact_bundle,
        generate_fn=generate_fn,
    )
    required_fields = {
        "case_id",
        "generated_text",
        "stop_reason",
        "latency_ms",
        "prompt_tokens",
        "completion_tokens",
        "model_version",
        "run_id",
        "tokenizer_id",
        "decoding",
        "generation_seed",
        "evaluator_results",
        "suite_version",
    }
    for record in results:
        assert required_fields <= record.keys()
        assert record["model_version"] == artifact_bundle.model_version
        assert record["run_id"] == artifact_bundle.run_id
        assert record["tokenizer_id"] == artifact_bundle.tokenizer_id
        assert record["suite_version"] == "v0.1"


def test_run_suite_rejects_incomplete_artifact_bundle(decoding_config):
    incomplete_bundle = ArtifactBundle(model_version="", run_id="run-1", tokenizer_id="tok-1")
    with pytest.raises(IncompatibleArtifactError):
        run_suite(
            DATASET_PATH,
            decoding_config=decoding_config,
            artifact_bundle=incomplete_bundle,
            generate_fn=DeterministicSeedEchoGenerator(),
        )


def test_run_suite_is_deterministic_given_same_inputs(decoding_config, artifact_bundle):
    generate_fn = DeterministicSeedEchoGenerator()
    first = run_suite(
        DATASET_PATH,
        decoding_config=decoding_config,
        artifact_bundle=artifact_bundle,
        generate_fn=generate_fn,
    )
    second = run_suite(
        DATASET_PATH,
        decoding_config=decoding_config,
        artifact_bundle=artifact_bundle,
        generate_fn=generate_fn,
    )

    def strip_latency(records):
        return [{k: v for k, v in r.items() if k != "latency_ms"} for r in records]

    assert strip_latency(first) == strip_latency(second)


def test_generation_seed_is_stable_per_case_across_runs(decoding_config, artifact_bundle):
    generate_fn = DeterministicSeedEchoGenerator()
    results = run_suite(
        DATASET_PATH,
        decoding_config=decoding_config,
        artifact_bundle=artifact_bundle,
        generate_fn=generate_fn,
    )
    for record in results:
        expected_seed = decoding_config.case_seed(record["case_id"])
        assert record["generation_seed"] == expected_seed


def test_broken_generator_is_caught_by_evaluators(decoding_config, artifact_bundle):
    results = run_suite(
        DATASET_PATH,
        decoding_config=decoding_config,
        artifact_bundle=artifact_bundle,
        generate_fn=BrokenGenerator(),
    )
    assert all(r["evaluator_results"]["overall_pass"] is False for r in results)
    assert all(r["evaluator_results"]["repetition"]["pass"] is False for r in results)


def test_summarize_results_counts_by_category_and_evaluator(decoding_config, artifact_bundle):
    cases = load_dataset(DATASET_PATH, suite_version="v0.1")
    generate_fn = canned_generator_for(cases)
    results = run_suite(
        DATASET_PATH,
        decoding_config=decoding_config,
        artifact_bundle=artifact_bundle,
        generate_fn=generate_fn,
    )
    summary = summarize_results(results)
    assert summary["total_cases"] == 10
    assert sum(bucket["total"] for bucket in summary["by_category"].values()) == 10
    assert "required_terms" in summary["by_evaluator"]


def test_write_results_jsonl_round_trips(tmp_path, decoding_config, artifact_bundle):
    cases = load_dataset(DATASET_PATH, suite_version="v0.1")
    generate_fn = canned_generator_for(cases)
    results = run_suite(
        DATASET_PATH,
        decoding_config=decoding_config,
        artifact_bundle=artifact_bundle,
        generate_fn=generate_fn,
    )
    out_path = tmp_path / "results.jsonl"
    write_results_jsonl(results, out_path)
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 10
    round_tripped = [json.loads(line) for line in lines]
    assert round_tripped[0]["case_id"] == results[0]["case_id"]


def test_write_summary_json_round_trips(tmp_path, decoding_config, artifact_bundle):
    cases = load_dataset(DATASET_PATH, suite_version="v0.1")
    generate_fn = canned_generator_for(cases)
    results = run_suite(
        DATASET_PATH,
        decoding_config=decoding_config,
        artifact_bundle=artifact_bundle,
        generate_fn=generate_fn,
    )
    summary = summarize_results(results)
    out_path = tmp_path / "summary.json"
    write_summary_json(summary, out_path)
    assert json.loads(out_path.read_text(encoding="utf-8"))["total_cases"] == 10


def test_output_dir_for_matches_plan_layout_plus_run_id():
    path = output_dir_for(
        Path("/tmp/out"), suite_version="v0.1", model_version="xlarge-plus", run_id="run-1"
    )
    assert path == Path("/tmp/out/v0.1/xlarge-plus/run-1")
