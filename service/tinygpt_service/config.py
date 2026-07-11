"""Service settings, sourced from environment variables on Cloud Run."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

_TRUE_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ServiceSettings:
    # Request guards
    max_message_chars: int = 2000
    max_body_bytes: int = 16_384
    rate_limit_per_minute: int = 30  # per client IP; 0 disables

    # Decoding defaults and ceilings (defaults match the baseline manifest's
    # generation_settings; the hard ceiling is enforced server-side)
    default_temperature: float = 0.7
    default_top_k: int = 30
    default_max_new_tokens: int = 200
    max_new_tokens_limit: int = 512

    # Generation is serialized; requests wait this long for the slot before 503
    generation_wait_seconds: float = 10.0

    # Tracing (fail startup when enabled but incomplete)
    tracing_enabled: bool = False
    arize_space_id: str | None = None
    arize_api_key: str | None = None
    arize_project_name: str = "tinygpt-chat"
    otlp_endpoint: str = "https://otlp.arize.com/v1/traces"

    # Provenance
    service_revision: str = "local"
    bundle_dir: str | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ServiceSettings":
        env = os.environ if env is None else env
        defaults = cls()

        def get_int(name: str, fallback: int) -> int:
            return int(env.get(name, fallback))

        def get_float(name: str, fallback: float) -> float:
            return float(env.get(name, fallback))

        return cls(
            max_message_chars=get_int("TINYGPT_MAX_MESSAGE_CHARS", defaults.max_message_chars),
            max_body_bytes=get_int("TINYGPT_MAX_BODY_BYTES", defaults.max_body_bytes),
            rate_limit_per_minute=get_int(
                "TINYGPT_RATE_LIMIT_PER_MINUTE", defaults.rate_limit_per_minute
            ),
            default_temperature=get_float(
                "TINYGPT_DEFAULT_TEMPERATURE", defaults.default_temperature
            ),
            default_top_k=get_int("TINYGPT_DEFAULT_TOP_K", defaults.default_top_k),
            default_max_new_tokens=get_int(
                "TINYGPT_DEFAULT_MAX_NEW_TOKENS", defaults.default_max_new_tokens
            ),
            max_new_tokens_limit=get_int(
                "TINYGPT_MAX_NEW_TOKENS_LIMIT", defaults.max_new_tokens_limit
            ),
            generation_wait_seconds=get_float(
                "TINYGPT_GENERATION_WAIT_SECONDS", defaults.generation_wait_seconds
            ),
            tracing_enabled=env.get("TINYGPT_TRACING_ENABLED", "").lower() in _TRUE_VALUES,
            arize_space_id=env.get("ARIZE_SPACE_ID") or None,
            arize_api_key=env.get("ARIZE_API_KEY") or None,
            arize_project_name=env.get("ARIZE_PROJECT_NAME", defaults.arize_project_name),
            otlp_endpoint=env.get("TINYGPT_OTLP_ENDPOINT", defaults.otlp_endpoint),
            # K_REVISION is set by Cloud Run
            service_revision=env.get("K_REVISION", defaults.service_revision),
            bundle_dir=env.get("TINYGPT_BUNDLE_DIR") or None,
        )
