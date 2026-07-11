"""The minimal contract the evaluation runner needs from a generation backend.

`core/` does not yet expose a stable inference API (it is being built
concurrently in a separate work stream). Rather than guess at that API, the
runner depends only on the small, explicit surface defined here: a
``GenerateFn`` callable that turns a ``GenerationRequest`` into a
``GenerationResult``. Anyone wiring up a real backend (``core/`` or otherwise)
only needs to satisfy this Protocol.

See ``evals/CORE_INTEGRATION_CONTRACT.md`` for the full write-up, including
what the runner deliberately does NOT ask of the backend (timing, artifact
identity) and why.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class GenerationRequest:
    """Everything a backend needs to produce one case's generation.

    ``seed`` is the per-case seed derived by
    ``DecodingConfig.case_seed(case_id)`` -- the backend must use it to make
    generation reproducible, however it chooses to (e.g. seeding a local RNG).
    """

    prompt: str
    max_new_tokens: int
    temperature: float
    top_k: int
    seed: int


@dataclass(frozen=True)
class GenerationResult:
    """What the runner needs back from a backend for one case.

    Deliberately excludes latency and artifact identity: the runner measures
    wall-clock latency itself around the call, and artifact identity
    (model_version/run_id/tokenizer_id) is supplied once per run via
    ``ArtifactBundle``, not repeated on every generation.
    """

    text: str
    stop_reason: str
    prompt_tokens: int
    completion_tokens: int


class GenerateFn(Protocol):
    def __call__(self, request: GenerationRequest) -> GenerationResult: ...


# Values the runner accepts for GenerationResult.stop_reason. Backends should
# map their own stop conditions onto this small, stable vocabulary so
# evaluators and reports don't need backend-specific branching.
STOP_REASONS = ("eos", "max_tokens", "stop_sequence", "other")


@dataclass(frozen=True)
class ArtifactBundle:
    """Identity of the checkpoint/tokenizer pair a run is evaluating.

    This is intentionally a flat, backend-agnostic record rather than a
    reference to a specific manifest shape from ``core/``, since that shape
    is not yet stable. Whoever wires up a real backend is responsible for
    mapping their manifest fields onto this bundle.
    """

    model_version: str
    run_id: str
    tokenizer_id: str

    def validate(self) -> None:
        missing = [
            field
            for field in ("model_version", "run_id", "tokenizer_id")
            if not getattr(self, field)
        ]
        if missing:
            raise IncompatibleArtifactError(
                f"artifact bundle is missing required identity fields: {missing}"
            )


class IncompatibleArtifactError(ValueError):
    """Raised when an artifact bundle lacks the identity fields the suite requires."""
