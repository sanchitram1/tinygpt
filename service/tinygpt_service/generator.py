"""The service-facing TinyGPT generator and artifact adapter."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from time import perf_counter
from typing import Iterator, Protocol, runtime_checkable

from .config import ServiceSettings


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


@dataclass(frozen=True)
class GenerationChunk:
    """One streamed text delta, optionally carrying the final result."""

    delta: str
    result: GenerationResult | None = None


@runtime_checkable
class StoryGenerator(Protocol):
    """What the service needs from the core runtime."""

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
    """Load one verified bundle and expose complete and streaming generation."""

    def __init__(self, bundle_dir: str, device: str = "auto") -> None:
        from core.artifacts import BundleError, load_bundle

        try:
            bundle = load_bundle(bundle_dir, device)
        except BundleError as exc:
            raise RuntimeError(f"unable to load TinyGPT model bundle: {exc}") from exc
        manifest = bundle.manifest
        model_meta = manifest["model"]
        self._model = bundle.model
        self._tokenizer = bundle.tokenizer
        self._device = bundle.device
        self._lock = threading.Lock()
        self._context_length = int(model_meta["context_length"])
        self._eos_id = self._tokenizer.token_to_id("<eos>")
        self.info = GeneratorInfo(
            model_version=str(manifest["model_version"]),
            run_id=str(manifest["run_id"]),
            tokenizer_id=str(manifest["tokenizer_id"]),
            device=str(self._device),
            context_length=self._context_length,
        )

    def stream(
        self,
        prompt: str,
        *,
        temperature: float,
        top_k: int,
        max_new_tokens: int,
    ) -> Iterator[GenerationChunk]:
        from core.generation import iter_generate_tokens

        started = perf_counter()
        prompt_ids = list(self._tokenizer.encode(prompt).ids)
        if not prompt_ids:
            bos_id = self._tokenizer.token_to_id("<bos>")
            if bos_id is None:
                raise ValueError("tokenizer has no <bos> token for an empty prompt")
            prompt_ids = [bos_id]
        if len(prompt_ids) > self._context_length:
            raise ValueError(
                f"prompt encodes to {len(prompt_ids)} tokens; context limit is "
                f"{self._context_length}"
            )

        generated_ids: list[int] = []
        previous_text = ""
        stop_reason = "max_new_tokens"
        with self._lock:
            for token_id in iter_generate_tokens(
                self._model,
                self._tokenizer,
                prompt,
                context_length=self._context_length,
                device=self._device,
                temperature=temperature,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
            ):
                generated_ids.append(token_id)
                current_text = self._tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )
                if current_text.startswith(previous_text):
                    delta = current_text[len(previous_text) :]
                else:
                    # A decoder should be monotonic, but do not silently lose
                    # output if a future tokenizer changes that assumption.
                    delta = current_text
                previous_text = current_text
                if self._eos_id is not None and token_id == self._eos_id:
                    stop_reason = "eos"
                if delta:
                    yield GenerationChunk(delta=delta)

        result = GenerationResult(
            text=previous_text,
            prompt_token_count=len(prompt_ids),
            output_token_count=len(generated_ids),
            stop_reason=stop_reason,
            latency_ms=(perf_counter() - started) * 1000,
        )
        yield GenerationChunk(delta="", result=result)

    def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        top_k: int,
        max_new_tokens: int,
    ) -> GenerationResult:
        result: GenerationResult | None = None
        for chunk in self.stream(
            prompt,
            temperature=temperature,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
        ):
            if chunk.result is not None:
                result = chunk.result
        if result is None:
            raise RuntimeError("TinyGPT generation ended without a result")
        return result


def load_generator(settings: ServiceSettings) -> StoryGenerator:
    """Load the configured model or raise loudly during application startup."""
    if not settings.bundle_dir:
        raise RuntimeError(
            "TINYGPT_BUNDLE_DIR is required; point it at a verified model bundle"
        )
    return TinyGPTGenerator(settings.bundle_dir, settings.device)
