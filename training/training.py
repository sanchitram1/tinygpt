import math
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from config import GlobalTrainingConfig, ModelConfig, RunConfig
from models import TinyGPT, TokenChunkDataset
from utils import count_parameters, make_dataloader


@torch.no_grad()
def evaluate(
    model: TinyGPT,
    data_loader: DataLoader,
    device: torch.device,
    vocab_size: int,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    losses = []
    total_tokens = 0
    autocast_dtype = torch.float16 if device.type == "cuda" else None
    for x, y in data_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=use_amp and device.type == "cuda",
        ):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        losses.append(loss.item())
        total_tokens += x.numel()
    mean_loss = float(np.mean(losses))
    model.train()
    return {
        "loss": mean_loss,
        "perplexity": float(math.exp(mean_loss)),
        "tokens_evaluated": total_tokens,
    }


def compute_lr(step: int, max_steps: int, warmup_steps: int, base_lr: float) -> float:
    if step <= warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    min_lr = 0.1 * base_lr
    return min_lr + (base_lr - min_lr) * cosine


def train_model(
    run_config: RunConfig,
    global_training_config: GlobalTrainingConfig,
    model_config: ModelConfig,
    tokenizer: Tokenizer,
    train_dataset: TokenChunkDataset,
    valid_dataset: TokenChunkDataset,
    device: torch.device,
) -> dict:
    if device.type == "cuda":
        torch.cuda.empty_cache()

    vocab_size = tokenizer.get_vocab_size()

    # init the model
    model = TinyGPT(
        vocab_size=vocab_size,
        context_length=global_training_config.context_length,
        d_model=model_config.d_model,
        n_heads=model_config.n_heads,
        n_layers=model_config.n_layers,
        d_ff=model_config.d_ff,
        dropout=model_config.dropout,
    ).to(device)

    # prep the data loaders
    train_loader = make_dataloader(
        train_dataset, batch_size=model_config.batch_size, shuffle=True
    )
    valid_loader = make_dataloader(
        valid_dataset, batch_size=model_config.batch_size, shuffle=False
    )

    # init the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=model_config.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=model_config.use_amp and device.type == "cuda"
    )

    # here's stuff that we'll save
    train_history: list[dict] = []
    valid_history: list[dict] = []
    train_iter = iter(train_loader)
    start_time = time.perf_counter()
    autocast_dtype = torch.float16 if device.type == "cuda" else None
    total_tokens_processed = 0

    # we'll save it here
    model_path = run_config.models / f"{model_config.name}.pt"

    # begin the training loop
    for step in range(1, model_config.max_steps + 1):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        # figure out the learning rate
        lr = compute_lr(
            step,
            model_config.max_steps,
            model_config.warmup_steps,
            model_config.learning_rate,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # calculate the loss
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=model_config.use_amp and device.type == "cuda",
        ):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), model_config.grad_clip_norm
            )
        )
        scaler.step(optimizer)
        scaler.update()

        batch_tokens = int(x.numel())
        total_tokens_processed += batch_tokens
        elapsed = time.perf_counter() - start_time
        tokens_per_second = total_tokens_processed / max(elapsed, 1e-9)
        train_history.append(
            {
                "step": step,
                "loss": float(loss.item()),
                "perplexity": float(math.exp(loss.item())),
                "lr": lr,
                "grad_norm": grad_norm,
                "tokens_per_second": tokens_per_second,
                "elapsed_seconds": elapsed,
            }
        )

        if (
            step == 1
            or step % global_training_config.checkpoint_every == 0
            or step == model_config.max_steps
        ):
            metrics = evaluate(
                model,
                valid_loader,
                device,
                vocab_size=vocab_size,
                use_amp=model_config.use_amp,
            )
            metrics["step"] = step
            valid_history.append(metrics)
            print(f"[{model_config.name}] Loss(step={step})={float(loss.item()):.2f}")

    total_time = time.perf_counter() - start_time
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(model_config),
            "vocab_size": vocab_size,
            "context_length": global_training_config.context_length,
        },
        model_path,
    )
    return {
        "config": asdict(model_config),
        "parameter_count": count_parameters(model),
        "train_history": train_history,
        "valid_history": valid_history,
        "model_path": str(model_path),
        "total_training_time_seconds": total_time,
        "tokens_processed": total_tokens_processed,
        "tokens_per_second": total_tokens_processed / max(total_time, 1e-9),
    }
