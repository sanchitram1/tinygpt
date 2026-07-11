import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from config import ModelConfig, RunConfig


class TokenChunkDataset(Dataset):
    def __init__(self, token_path: Path, total_tokens: int, context_length: int):
        self.tokens = np.memmap(
            token_path, dtype=np.uint32, mode="r", shape=(total_tokens,)
        )
        self.context_length = context_length
        self.num_samples = (total_tokens - 1) // context_length

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns a tensor (input, target), like ((once, upon), (upon, a))"""
        start = idx * self.context_length
        end = start + self.context_length
        x = np.asarray(self.tokens[start:end], dtype=np.int64)
        y = np.asarray(self.tokens[start + 1 : end + 1], dtype=np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        """A torch module to do sinusoidal positional encoding"""
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
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        """A single transformer block"""
        super().__init__()

        # guard
        assert d_model % n_heads == 0

        # configuration values
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # feedforward is a single hidden layer
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

        # two normalization layers
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # configure dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # residual is the original input, which we'll add to softmax output later
        residual = x

        # normalize
        x = self.ln1(x)

        # batch size, sequence length, and model dimension is the input tensor
        bsz, seq_len, d_model = x.shape

        # output dimension is (bsz, 3, seq_len, d_model)
        qkv = self.qkv(x).view(bsz, seq_len, 3, self.n_heads, self.head_dim)

        # extract the q, k, v portions and transpose them
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # attention requires a very specific shape, which is ready now
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)

        # here we map the output back to the dimension we expect
        attn = attn.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)

        # add the residual input back
        x = residual + self.dropout(self.out_proj(attn))

        # dropout for the FFNN
        x = x + self.dropout(self.ff(self.ln2(x)))

        return x


class TinyGPT(nn.Module):
    """Our TinyStory GPT!"""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, context_length)

        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout
                )
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
        """Performs the forward pass"""
        # first embed the tokens
        x = self.token_embedding(input_ids)

        # then encode them
        x = self.pos_encoding(x)

        # do a dropout
        x = self.dropout(x)

        # pass each through the transformer blocks
        for block in self.blocks:
            x = block(x)

        # and do a normalization
        x = self.final_ln(x)
        return self.lm_head(x)


def model_checkpoint_path(run_config: RunConfig, model_config: ModelConfig) -> Path:
    return run_config.models / f"{model_config.name}.pt"


def load_model(model_path: Path, device: torch.device, train=False) -> TinyGPT:
    """Loads a saved TinyGPT checkpoint and returns it in eval mode."""
    checkpoint = torch.load(model_path, map_location=device)
    config = ModelConfig(**checkpoint["config"])
    model = TinyGPT(
        vocab_size=checkpoint["vocab_size"],
        context_length=checkpoint["context_length"],
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        dropout=config.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    if train:
        model.train()
    else:
        model.eval()

    return model
