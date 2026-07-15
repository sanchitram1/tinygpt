"""Request/response schemas for the chat API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# Absolute ceilings; the effective (usually tighter) limits come from
# ServiceSettings and are enforced in the route.
_HARD_MESSAGE_CEILING = 8000
_SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=_HARD_MESSAGE_CEILING)
    session_id: str | None = Field(default=None, pattern=_SESSION_ID_PATTERN)
    temperature: float | None = Field(default=None, gt=0, le=2)
    top_k: int | None = Field(default=None, ge=1, le=1000)
    max_new_tokens: int | None = Field(default=None, ge=1)
    synthetic: bool = False


class Decoding(BaseModel):
    temperature: float
    top_k: int
    max_new_tokens: int


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    request_id: str
    model_version: str
    run_id: str
    tokenizer_id: str
    prompt_token_count: int
    output_token_count: int
    stop_reason: str
    latency_ms: float
    decoding: Decoding
    synthetic: bool
