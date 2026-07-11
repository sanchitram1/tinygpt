"""HTTP routes: chat UI, health/readiness, and the generation endpoint."""

from __future__ import annotations

import asyncio
import functools
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse

from .config import ServiceSettings
from .errors import ServiceError, error_response
from .generator import StoryGenerator
from .schemas import ChatRequest, ChatResponse, Decoding
from .tracing import build_chat_trace_attributes

STATIC_DIR = Path(__file__).parent / "static"

router = APIRouter()


@router.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@router.get("/healthz")
async def healthz(request: Request) -> dict[str, str]:
    """Process liveness; never loads the model or generates."""
    settings: ServiceSettings = request.app.state.settings
    return {"status": "ok", "revision": settings.service_revision}


@router.get("/readyz")
async def readyz(request: Request) -> JSONResponse:
    """Model/tokenizer readiness after artifact validation."""
    generator: StoryGenerator | None = request.app.state.generator
    if generator is None:
        return error_response(
            503,
            "model_unavailable",
            "model artifact is not loaded",
            getattr(request.state, "request_id", None),
        )
    info = generator.info
    return JSONResponse(
        {
            "status": "ready",
            "model_version": info.model_version,
            "run_id": info.run_id,
            "tokenizer_id": info.tokenizer_id,
            "device": info.device,
        }
    )


def _resolve_decoding(payload: ChatRequest, settings: ServiceSettings) -> dict[str, Any]:
    requested = payload.max_new_tokens or settings.default_max_new_tokens
    return {
        "temperature": payload.temperature or settings.default_temperature,
        "top_k": payload.top_k or settings.default_top_k,
        "max_new_tokens": min(requested, settings.max_new_tokens_limit),
    }


@router.post("/api/chat", response_model=ChatResponse)
async def chat(request: Request, payload: ChatRequest) -> ChatResponse:
    state = request.app.state
    settings: ServiceSettings = state.settings
    request_id: str = getattr(request.state, "request_id", str(uuid.uuid4()))

    client_key = request.client.host if request.client else "unknown"
    if not state.rate_limiter.allow(client_key):
        raise ServiceError(429, "rate_limited", "too many requests; slow down")

    if len(payload.message) > settings.max_message_chars:
        raise ServiceError(
            422,
            "invalid_request",
            f"message exceeds {settings.max_message_chars} characters",
        )

    session_id = payload.session_id or str(uuid.uuid4())
    decoding = _resolve_decoding(payload, settings)
    generator: StoryGenerator | None = state.generator

    def record(http_status: int, result=None, error_type: str | None = None) -> None:
        state.trace_sink.record_generation(
            build_chat_trace_attributes(
                message=payload.message,
                session_id=session_id,
                request_id=request_id,
                synthetic=payload.synthetic,
                decoding=decoding,
                service_revision=settings.service_revision,
                http_status=http_status,
                info=generator.info if generator is not None else None,
                result=result,
                error_type=error_type,
            )
        )

    if generator is None:
        record(503, error_type="model_unavailable")
        raise ServiceError(503, "model_unavailable", "model artifact is not loaded")

    acquired = await asyncio.to_thread(
        functools.partial(
            state.generation_semaphore.acquire,
            timeout=settings.generation_wait_seconds,
        )
    )
    if not acquired:
        record(503, error_type="model_busy")
        raise ServiceError(503, "model_busy", "generation is busy; retry shortly")

    try:
        result = await asyncio.to_thread(
            functools.partial(generator.generate, payload.message, **decoding)
        )
    except Exception as exc:
        record(500, error_type=type(exc).__name__)
        raise ServiceError(500, "generation_failed", "story generation failed") from exc
    finally:
        state.generation_semaphore.release()

    record(200, result=result)
    info = generator.info
    return ChatResponse(
        reply=result.text,
        session_id=session_id,
        request_id=request_id,
        model_version=info.model_version,
        run_id=info.run_id,
        tokenizer_id=info.tokenizer_id,
        prompt_token_count=result.prompt_token_count,
        output_token_count=result.output_token_count,
        stop_reason=result.stop_reason,
        latency_ms=result.latency_ms,
        decoding=Decoding(**decoding),
        synthetic=payload.synthetic,
    )
