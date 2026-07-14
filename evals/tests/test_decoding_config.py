from pathlib import Path

import pytest

from eval_suite.decoding_config import (
    DecodingConfigError,
    load_decoding_config,
    validate_decoding_config,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "decoding-v0.1.json"


def test_loads_released_decoding_config():
    config = load_decoding_config(CONFIG_PATH)
    assert config.version == "v0.1"
    assert config.temperature > 0
    assert config.top_k >= 1
    assert config.max_new_tokens >= 1


def test_case_seed_is_deterministic():
    config = load_decoding_config(CONFIG_PATH)
    assert config.case_seed("case-a") == config.case_seed("case-a")


def test_case_seed_differs_across_cases():
    config = load_decoding_config(CONFIG_PATH)
    assert config.case_seed("case-a") != config.case_seed("case-b")


def test_case_seed_differs_across_suite_seeds():
    from eval_suite.decoding_config import DecodingConfig

    config_a = DecodingConfig(version="v0.1", temperature=0.7, top_k=30, max_new_tokens=200, suite_seed=1)
    config_b = DecodingConfig(version="v0.1", temperature=0.7, top_k=30, max_new_tokens=200, suite_seed=2)
    assert config_a.case_seed("case-a") != config_b.case_seed("case-a")


def test_rejects_non_positive_temperature():
    with pytest.raises(DecodingConfigError):
        validate_decoding_config(
            {
                "version": "v0.1",
                "temperature": 0,
                "top_k": 30,
                "max_new_tokens": 200,
                "suite_seed": 1,
            }
        )


def test_rejects_missing_field():
    with pytest.raises(DecodingConfigError):
        validate_decoding_config(
            {
                "version": "v0.1",
                "temperature": 0.7,
                "top_k": 30,
                "suite_seed": 1,
            }
        )


def test_rejects_bad_version_format():
    with pytest.raises(DecodingConfigError):
        validate_decoding_config(
            {
                "version": "0.1",
                "temperature": 0.7,
                "top_k": 30,
                "max_new_tokens": 200,
                "suite_seed": 1,
            }
        )
