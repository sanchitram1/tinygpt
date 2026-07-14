"""The TinyGPT model definition and checkpoint loading helpers.

This module intentionally has no dependency on the training package so the
service can load a model in a small serving image.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ln1(x)
        bsz, seq_len, d_model = x.shape
        qkv = self.qkv(x).view(bsz, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        attn = attn.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
        x = residual + self.dropout(self.out_proj(attn))
        return x + self.dropout(self.ff(self.ln2(x)))


class TinyGPT(nn.Module):
    """The TinyStories GPT architecture used by the recovered checkpoints."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, context_length)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )
        self.final_ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(input_ids)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_ln(x))


def model_from_checkpoint(
    checkpoint: Mapping[str, Any], device: torch.device
) -> TinyGPT:
    """Construct and load a TinyGPT from the established checkpoint shape."""
    required = {"model_state", "config", "vocab_size", "context_length"}
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"checkpoint is missing required fields: {', '.join(missing)}")
    config = checkpoint["config"]
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint config must be an object")
    try:
        model = TinyGPT(
            vocab_size=int(checkpoint["vocab_size"]),
            context_length=int(checkpoint["context_length"]),
            d_model=int(config["d_model"]),
            n_heads=int(config["n_heads"]),
            n_layers=int(config["n_layers"]),
            d_ff=int(config["d_ff"]),
            dropout=float(config.get("dropout", 0.1)),
        ).to(device)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("checkpoint contains an invalid model configuration") from exc
    try:
        model.load_state_dict(checkpoint["model_state"])
    except (RuntimeError, TypeError) as exc:
        raise ValueError("checkpoint weights do not match the model configuration") from exc
    model.eval()
    return model


def load_checkpoint(path: Path, device: torch.device) -> tuple[TinyGPT, Mapping[str, Any]]:
    """Load a trusted local checkpoint and return the model plus raw metadata."""
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except Exception as exc:
        raise ValueError(f"unable to load checkpoint {path}") from exc
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must contain a mapping")
    return model_from_checkpoint(checkpoint, device), checkpoint
