"""Inference-time token sampling for TinyGPT."""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn.functional as F

from .model import TinyGPT


def top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.shape[-1]:
        return logits
    values, _ = torch.topk(logits, top_k, dim=-1)
    cutoff = values[..., -1, None]
    return torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)


def iter_generate_tokens(
    model: TinyGPT,
    tokenizer,
    prompt: str,
    *,
    context_length: int,
    device: torch.device,
    temperature: float,
    top_k: int,
    max_new_tokens: int,
) -> Iterator[int]:
    """Yield sampled token IDs, stopping at EOS or the output ceiling."""
    prompt_ids = list(tokenizer.encode(prompt).ids)
    if not prompt_ids:
        bos_id = tokenizer.token_to_id("<bos>")
        if bos_id is None:
            raise ValueError("tokenizer has no <bos> token for an empty prompt")
        prompt_ids = [bos_id]
    if len(prompt_ids) > context_length:
        raise ValueError(
            f"prompt encodes to {len(prompt_ids)} tokens; context limit is {context_length}"
        )

    eos_id = tokenizer.token_to_id("<eos>")
    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            input_tokens = tokens[:, -context_length:]
            logits = model(input_tokens)[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(top_k_filter(logits, top_k), dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = int(next_token.item())
            tokens = torch.cat([tokens, next_token], dim=1)
            yield token_id
            if eos_id is not None and token_id == eos_id:
                return
