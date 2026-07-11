import torch.nn as nn
from fine_tuning.lora import LoRALinear

from models import TinyGPT


def apply_lora_to_model(
    model: TinyGPT,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    target_ff: bool = False,
    target_layers: int | None = None,
) -> TinyGPT:
    """Apply LoRA to selected layers of the model.

    Args:
        target_layers: If set, only apply LoRA to the LAST N transformer blocks.
                       If None (default), apply to all blocks.
    """
    blocks = model.blocks
    if target_layers is not None:
        blocks = blocks[-target_layers:]

    for block in blocks:
        block.qkv = LoRALinear(block.qkv, rank=rank, alpha=alpha, dropout=dropout)
        block.out_proj = LoRALinear(
            block.out_proj, rank=rank, alpha=alpha, dropout=dropout
        )

        if target_ff:
            block.ff[0] = LoRALinear(
                block.ff[0], rank=rank, alpha=alpha, dropout=dropout
            )
            block.ff[2] = LoRALinear(
                block.ff[2], rank=rank, alpha=alpha, dropout=dropout
            )
    return model


def freeze_non_lora_parameters(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name
