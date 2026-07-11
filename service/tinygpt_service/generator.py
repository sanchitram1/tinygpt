"""Story-generation interface the service depends on.

The Plan 1 core runtime API is still in flux, so nothing here imports from
``core/``. The service programs against the small ``StoryGenerator`` protocol
below; tests use a fake, and the real TinyGPT adapter is an explicit
integration TODO (see ``TinyGPTGenerator``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .config import ServiceSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeneratorInfo:
    """Provenance for the loaded model, sourced from the artifact manifest."""

    model_version: str
    run_id: str
    tokenizer_id: str
    device: str
    context_length: int


@dataclass(frozen=True)
class GenerationResult:
    text: str
    prompt_token_count: int
    output_token_count: int
    stop_reason: str  # e.g. "eos" | "max_new_tokens"
    latency_ms: float


@runtime_checkable
class StoryGenerator(Protocol):
    """What the service needs from the core runtime.

    Implementations must be safe to call from a worker thread, enforce the
    model context budget (truncate or reject prompts that exceed it), and
    already be in eval mode on the target device.
    """

    info: GeneratorInfo

    def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        top_k: int,
        max_new_tokens: int,
    ) -> GenerationResult: ...


class TinyGPTGenerator:
    """TODO(plan-1 integration): real adapter over the core TinyGPT runtime.

    Once the Plan 1 core API stabilizes, this adapter should:
      1. Read checkpoint, tokenizer, and ``manifest.json`` from one artifact
         bundle directory (``settings.bundle_dir``); never a bare checkpoint.
      2. Verify model/tokenizer digests against the manifest and fail loudly
         on mismatch.
      3. Load the model once, move it to the serving device, call ``eval()``.
      4. Populate ``GeneratorInfo`` from the manifest (model_version, run_id,
         tokenizer_id, device, context_length).
      5. Implement ``generate`` returning text, prompt/output token counts,
         stop reason, and wall-time latency, enforcing the context budget.
    """

    def __init__(self, bundle_dir: str) -> None:
        raise NotImplementedError(
            "TinyGPT core adapter is not integrated yet; the Plan 1 core "
            "runtime API is not final. See TinyGPTGenerator docstring for "
            "the required core interface."
        )


def load_generator(settings: ServiceSettings) -> StoryGenerator | None:
    """Return the real generator once integrated; ``None`` keeps the service
    up but not-ready (``/readyz`` and ``/api/chat`` return 503)."""
    # TODO(plan-1 integration): return TinyGPTGenerator(settings.bundle_dir)
    logger.warning(
        "TinyGPT adapter not integrated; serving in not-ready mode "
        "(bundle_dir=%s)",
        settings.bundle_dir,
    )
    return None
