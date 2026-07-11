import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset

from config import GlobalTrainingConfig, TokenConfig
from models import TinyGPT


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def make_dataloader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        drop_last=shuffle,
    )


def save_json(data: dict, output_path: Path) -> None:
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.numel():
        return logits
    values, _ = torch.topk(logits, top_k)
    cutoff = values[..., -1, None]
    return torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)


@torch.no_grad()
def generate_text(
    token_config: TokenConfig,
    global_training_config: GlobalTrainingConfig,
    model: TinyGPT,
    tokenizer: Tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 120,
    temperature: float = 0.9,
    top_k: int = 40,
) -> str:
    ids = tokenizer.encode(prompt).ids
    if not ids:
        ids = [tokenizer.token_to_id(token_config.bos)]

    eos_id = tokenizer.token_to_id(token_config.eos)
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        input_tokens = tokens[:, -global_training_config.context_length :]
        logits = model(input_tokens)[:, -1, :] / max(temperature, 1e-5)
        logits = top_k_filter(logits, top_k=top_k)
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)
        if eos_id is not None and int(next_token.item()) == eos_id:
            break
    decoded = tokenizer.decode(tokens[0].tolist(), skip_special_tokens=True)
    return decoded
