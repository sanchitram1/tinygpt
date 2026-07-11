"""Trace-attribute construction and telemetry configuration.

Attribute construction is a pure function so it can be tested without any
OpenTelemetry dependency. The OTLP/Arize exporter is only initialized when
tracing is explicitly enabled, and startup fails clearly when tracing is
enabled but required configuration is missing.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping, Protocol

from .config import ServiceSettings
from .generator import GenerationResult, GeneratorInfo

logger = logging.getLogger(__name__)

# The nine attributes Plan 3 requires on every generation trace.
REQUIRED_TRACE_ATTRIBUTES = (
    "model_version",
    "run_id",
    "tokenizer_id",
    "decoding_parameters",
    "prompt_token_count",
    "output_token_count",
    "latency_ms",
    "session_id",
    "synthetic",
)


class TelemetryConfigError(RuntimeError):
    """Tracing is enabled but required configuration is missing."""


def validate_tracing_config(settings: ServiceSettings) -> None:
    if not settings.tracing_enabled:
        return
    missing = [
        name
        for name, value in (
            ("ARIZE_SPACE_ID", settings.arize_space_id),
            ("ARIZE_API_KEY", settings.arize_api_key),
            ("ARIZE_PROJECT_NAME", settings.arize_project_name),
            ("TINYGPT_OTLP_ENDPOINT", settings.otlp_endpoint),
        )
        if not value
    ]
    if missing:
        raise TelemetryConfigError(
            "tracing is enabled (TINYGPT_TRACING_ENABLED) but required "
            f"configuration is missing: {', '.join(missing)}. Set the missing "
            "values or disable tracing for local development."
        )


def build_chat_trace_attributes(
    *,
    message: str,
    session_id: str,
    request_id: str,
    synthetic: bool,
    decoding: Mapping[str, Any],
    service_revision: str,
    http_status: int,
    info: GeneratorInfo | None,
    result: GenerationResult | None = None,
    error_type: str | None = None,
) -> dict[str, Any]:
    """Build the flat attribute set for one generation span.

    Works for failed generations too (``result=None`` with ``error_type``).
    Never include credentials or personal identifiers here; the session ID is
    a random UUID or a client-chosen opaque token.
    """
    attributes: dict[str, Any] = {
        # OpenInference-compatible input/output attributes
        "openinference.span.kind": "LLM",
        "input.value": message,
        "llm.invocation_parameters": json.dumps(dict(decoding)),
        # Project-required attributes (REQUIRED_TRACE_ATTRIBUTES)
        "model_version": info.model_version if info else "unavailable",
        "run_id": info.run_id if info else "unavailable",
        "tokenizer_id": info.tokenizer_id if info else "unavailable",
        "decoding_parameters": json.dumps(dict(decoding)),
        "prompt_token_count": result.prompt_token_count if result else 0,
        "output_token_count": result.output_token_count if result else 0,
        "latency_ms": result.latency_ms if result else 0.0,
        "session_id": session_id,
        "synthetic": synthetic,
        # Operational context
        "service.revision": service_revision,
        "http.status_code": http_status,
        "model.device": info.device if info else "unavailable",
        "request.id": request_id,
    }
    if result is not None:
        attributes["output.value"] = result.text
        attributes["stop_reason"] = result.stop_reason
    if error_type is not None:
        attributes["error.type"] = error_type
    return attributes


class TraceSink(Protocol):
    """Destination for generation trace attributes."""

    def record_generation(self, attributes: Mapping[str, Any]) -> None: ...


class NullTraceSink:
    """Local development mode: tracing explicitly disabled."""

    def record_generation(self, attributes: Mapping[str, Any]) -> None:
        return None


class OtelTraceSink:
    """Exports one request span with a child generation span over OTLP.

    OpenTelemetry is imported lazily so the base install never requires it.
    """

    def __init__(self, tracer: Any) -> None:
        self._tracer = tracer

    @classmethod
    def create(cls, settings: ServiceSettings) -> "OtelTraceSink":
        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError as exc:
            raise TelemetryConfigError(
                "tracing is enabled but OpenTelemetry is not installed; "
                "install the service with the [tracing] extra."
            ) from exc

        exporter = OTLPSpanExporter(
            endpoint=settings.otlp_endpoint,
            headers={
                "space_id": settings.arize_space_id or "",
                "api_key": settings.arize_api_key or "",
            },
        )
        provider = TracerProvider(
            resource=Resource.create({"model_id": settings.arize_project_name})
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        return cls(trace.get_tracer("tinygpt-service"))

    def record_generation(self, attributes: Mapping[str, Any]) -> None:
        # TODO(integration): open these spans around the live request/generation
        # so span durations match wall time, instead of recording after the fact.
        with self._tracer.start_as_current_span("POST /api/chat") as request_span:
            request_span.set_attribute("request.id", str(attributes.get("request.id")))
            request_span.set_attribute(
                "http.status_code", int(attributes.get("http.status_code", 0))
            )
            with self._tracer.start_as_current_span("tinygpt.generate") as span:
                for key, value in attributes.items():
                    span.set_attribute(key, value)


def create_trace_sink(settings: ServiceSettings) -> TraceSink:
    """Fail fast on bad config; return a no-op sink when tracing is disabled."""
    validate_tracing_config(settings)
    if not settings.tracing_enabled:
        logger.info("tracing disabled; generation traces will not be exported")
        return NullTraceSink()
    return OtelTraceSink.create(settings)
