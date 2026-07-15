import json

import pytest

from conftest import FAKE_INFO
from tinygpt_service.config import ServiceSettings
from tinygpt_service.generator import GenerationResult
from tinygpt_service.main import create_app
from tinygpt_service.tracing import (
    REQUIRED_TRACE_ATTRIBUTES,
    NullTraceSink,
    TelemetryConfigError,
    build_chat_trace_attributes,
    create_trace_sink,
    validate_tracing_config,
)

DECODING = {"temperature": 0.7, "top_k": 30, "max_new_tokens": 200}


def _success_attributes():
    return build_chat_trace_attributes(
        message="a brave fox",
        session_id="session-1",
        request_id="req-1",
        synthetic=False,
        decoding=DECODING,
        service_revision="rev-42",
        http_status=200,
        info=FAKE_INFO,
        result=GenerationResult(
            text="Once upon a time...",
            prompt_token_count=3,
            output_token_count=120,
            stop_reason="eos",
            latency_ms=812.5,
        ),
    )


def test_success_attributes_include_all_nine_required():
    attributes = _success_attributes()
    for name in REQUIRED_TRACE_ATTRIBUTES:
        assert name in attributes, name
    assert attributes["model_version"] == "xlarge-plus"
    assert attributes["run_id"] == "test-run-001"
    assert attributes["tokenizer_id"] == "tinystories_bpe_5000"
    assert json.loads(attributes["decoding_parameters"]) == DECODING
    assert attributes["prompt_token_count"] == 3
    assert attributes["output_token_count"] == 120
    assert attributes["latency_ms"] == 812.5
    assert attributes["session_id"] == "session-1"
    assert attributes["synthetic"] is False


def test_success_attributes_include_openinference_and_context():
    attributes = _success_attributes()
    assert attributes["openinference.span.kind"] == "LLM"
    assert attributes["input.value"] == "a brave fox"
    assert attributes["output.value"] == "Once upon a time..."
    assert attributes["service.revision"] == "rev-42"
    assert attributes["http.status_code"] == 200
    assert attributes["model.device"] == "cpu"
    assert attributes["request.id"] == "req-1"
    assert "error.type" not in attributes


def test_failed_generation_attributes():
    attributes = build_chat_trace_attributes(
        message="hi",
        session_id="session-1",
        request_id="req-2",
        synthetic=True,
        decoding=DECODING,
        service_revision="rev-42",
        http_status=503,
        info=None,
        error_type="model_unavailable",
    )
    for name in REQUIRED_TRACE_ATTRIBUTES:
        assert name in attributes, name
    assert attributes["error.type"] == "model_unavailable"
    assert attributes["model_version"] == "unavailable"
    assert attributes["output_token_count"] == 0
    assert attributes["synthetic"] is True
    assert "output.value" not in attributes


def test_no_credential_shaped_attributes():
    for key in _success_attributes():
        assert "key" not in key.lower()
        assert "secret" not in key.lower()
        assert "authorization" not in key.lower()


def test_validate_passes_when_tracing_disabled():
    validate_tracing_config(ServiceSettings(tracing_enabled=False))


def test_validate_fails_listing_missing_values():
    settings = ServiceSettings(tracing_enabled=True, arize_space_id="space-1")
    with pytest.raises(TelemetryConfigError) as excinfo:
        validate_tracing_config(settings)
    assert "ARIZE_API_KEY" in str(excinfo.value)
    assert "ARIZE_SPACE_ID" not in str(excinfo.value)


def test_create_trace_sink_disabled_is_noop():
    sink = create_trace_sink(ServiceSettings(tracing_enabled=False))
    assert isinstance(sink, NullTraceSink)
    sink.record_generation({"any": "thing"})  # must not raise


def test_create_app_fails_fast_on_bad_tracing_config():
    with pytest.raises(TelemetryConfigError):
        create_app(settings=ServiceSettings(tracing_enabled=True), generator=None)
