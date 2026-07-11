from __future__ import annotations

from typing import Any, Mapping

import pytest
from fastapi.testclient import TestClient

from tinygpt_service.config import ServiceSettings
from tinygpt_service.generator import GenerationResult, GeneratorInfo
from tinygpt_service.main import create_app

FAKE_INFO = GeneratorInfo(
    model_version="xlarge-plus",
    run_id="test-run-001",
    tokenizer_id="tinystories_bpe_5000",
    device="cpu",
    context_length=512,
)


class FakeStoryGenerator:
    """Deterministic stand-in for the not-yet-integrated TinyGPT adapter."""

    info = FAKE_INFO

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(
        self, prompt: str, *, temperature: float, top_k: int, max_new_tokens: int
    ) -> GenerationResult:
        self.calls.append(
            {
                "prompt": prompt,
                "temperature": temperature,
                "top_k": top_k,
                "max_new_tokens": max_new_tokens,
            }
        )
        return GenerationResult(
            text=f"Once upon a time, {prompt}",
            prompt_token_count=len(prompt.split()),
            output_token_count=5,
            stop_reason="eos",
            latency_ms=1.5,
        )


class FailingGenerator:
    info = FAKE_INFO

    def generate(self, prompt, *, temperature, top_k, max_new_tokens):
        raise RuntimeError("cuda exploded")


class CaptureTraceSink:
    def __init__(self) -> None:
        self.records: list[Mapping[str, Any]] = []

    def record_generation(self, attributes: Mapping[str, Any]) -> None:
        self.records.append(dict(attributes))


@pytest.fixture
def make_client():
    def _make(generator=None, sink=None, **settings_overrides) -> TestClient:
        settings = ServiceSettings(**settings_overrides)
        app = create_app(
            settings=settings,
            generator=generator,
            trace_sink=sink or CaptureTraceSink(),
        )
        client = TestClient(app, raise_server_exceptions=False)
        client.app_state = app.state
        return client

    return _make


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client(generator=FakeStoryGenerator())
