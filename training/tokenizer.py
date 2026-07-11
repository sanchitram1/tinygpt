from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from config import SHARED_DIR, TokenConfig, TokenizationConfig


def _tokenizer_path(tokenization_config: TokenizationConfig) -> Path:
    return (
        SHARED_DIR
        / "tokenizers"
        / f"tinystories_bpe_metaspace_{tokenization_config.vocab_size}_{tokenization_config.max_train_stories}.json"
    )


def _memmap_path(tokenization_config: TokenizationConfig, input_path: Path) -> Path:
    name = input_path.name.lower()

    if "instruction" in name and "valid" in name:
        split = "instruction_valid"
    elif "instruction" in name and "train" in name:
        split = "instruction_train"
    elif "valid" in name:
        split = "valid"
    else:
        split = "train"

    return (
        SHARED_DIR
        / "memmaps"
        / f"{split}_tokens_metaspace_{tokenization_config.max_train_stories}_v{tokenization_config.vocab_size}.bin"
    )


def iter_stories(
    tokenization_config: TokenizationConfig,
    training_file_path: Path,
) -> Iterable[str]:
    """Yields stories from the txt file one at a time"""
    buffer = ""
    count = 0
    with training_file_path.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            buffer += chunk
            pieces = buffer.split(tokenization_config.story_delimiter)
            buffer = pieces.pop()
            for piece in pieces:
                story = piece.strip()
                if not story:
                    continue
                yield story
                count += 1
                if (
                    tokenization_config.max_train_stories is not None
                    and count >= tokenization_config.max_train_stories
                ):
                    return
        tail = buffer.strip()
        if tail and (
            tokenization_config.max_train_stories is None
            or count < tokenization_config.max_train_stories
        ):
            yield tail


def count_tokens(
    tokenization_config: TokenizationConfig,
    tokenizer: Tokenizer,
    training_file_path: Path,
) -> int:
    total = 0
    for story in iter_stories(tokenization_config, training_file_path):
        total += len(tokenizer.encode(story).ids) + 1  # +1 for the EOS Token!
    return total


def build_token_memmap(
    tokenization_config: TokenizationConfig,
    token_config: TokenConfig,
    tokenizer: Tokenizer,
    path: Path,
    total_tokens: int,
    output_path: Path | None = None,
) -> Path:
    """Returns the path of the built token memmap

    A memmap is basically like a token stream, but optimized for quick retrieval"""
    output_path = (
        Path(output_path) if output_path else _memmap_path(tokenization_config, path)
    )
    expected_bytes = total_tokens * np.dtype(np.uint32).itemsize

    if output_path.exists():
        actual_bytes = output_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"Found existing memmap at {output_path}, but its size is "
                f"{actual_bytes} bytes; expected {expected_bytes} bytes for "
                f"{total_tokens} uint32 tokens."
            )
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # grab the EOS ID
    eos_id = tokenizer.token_to_id(token_config.eos)
    assert eos_id is not None

    # build the memmap, defining the shape up front
    token_array = np.memmap(
        output_path, dtype=np.uint32, mode="w+", shape=(total_tokens,)
    )

    # iterate through the dataset and build it as a stream of tokens
    offset = 0
    for story in iter_stories(tokenization_config, path):
        ids = tokenizer.encode(story).ids

        # force a gap here
        next_offset = offset + len(ids) + 1
        if next_offset > total_tokens:
            raise ValueError(
                f"Memmap token count is too small: need at least {next_offset} "
                f"tokens, but total_tokens={total_tokens}."
            )
        token_array[offset : offset + len(ids)] = ids

        # put the EOS token in the gap
        token_array[offset + len(ids)] = eos_id
        offset = next_offset
    token_array.flush()
    if offset != total_tokens:
        raise ValueError(
            f"Memmap token count mismatch: wrote {offset} tokens, but "
            f"total_tokens={total_tokens}."
        )

    return output_path


def build_tokenizer(
    tokenization_config: TokenizationConfig,
    token_config: TokenConfig,
    train_path: Path,
) -> Tokenizer:
    """Trains a tokenizer based on the inputted data"""
    tokenizer_path = _tokenizer_path(tokenization_config)
    if tokenizer_path.exists():
        print(f"Found it in {tokenizer_path}, loading from there")
        return Tokenizer.from_file(str(tokenizer_path))

    # initialize
    tokenizer = Tokenizer(models.BPE(unk_token=token_config.unk))

    # define the metaspace and decoders
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement="▁", prepend_scheme="always"
    )
    tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")

    # init the trainer
    trainer = trainers.BpeTrainer(
        vocab_size=tokenization_config.vocab_size,
        min_frequency=2,
        show_progress=False,
        special_tokens=[
            token_config.bos,
            token_config.eos,
            token_config.pad,
            token_config.unk,
        ],
    )

    # train!
    tokenizer.train_from_iterator(
        iter_stories(tokenization_config, train_path),
        trainer=trainer,
        length=tokenization_config.max_train_stories,
    )
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_path))
    return tokenizer


def get_token_vector(
    tokenizer: Tokenizer,
    embedding_source: torch.nn.Module | torch.Tensor,
    token_text: str,
) -> torch.Tensor:
    """Returns the learned embedding vector for one tokenizer token."""
    token_id = tokenizer.token_to_id(token_text)
    if token_id is None:
        raise ValueError(f"Token {token_text!r} is not in the tokenizer vocabulary.")

    embeddings = _embedding_matrix(embedding_source)
    return embeddings[token_id].detach().clone()


def find_closest_tokens(
    tokenizer: Tokenizer,
    embedding_source: torch.nn.Module | torch.Tensor,
    target_vector: torch.Tensor,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Finds tokenizer tokens closest to a vector by cosine similarity."""
    embeddings = _embedding_matrix(embedding_source)
    target = target_vector.detach().to(embeddings.device, dtype=embeddings.dtype)
    if target.ndim == 1:
        target = target.unsqueeze(0)

    normalized_embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    normalized_target = torch.nn.functional.normalize(target, dim=1)
    similarities = normalized_target @ normalized_embeddings.T
    values, indices = torch.topk(similarities.squeeze(0), k=top_k)

    return [
        (tokenizer.id_to_token(int(index)), float(value))
        for value, index in zip(values.detach().cpu(), indices.detach().cpu())
    ]


def _embedding_matrix(embedding_source: torch.nn.Module | torch.Tensor) -> torch.Tensor:
    if isinstance(embedding_source, torch.Tensor):
        return embedding_source.detach()
    if hasattr(embedding_source, "token_embedding"):
        return embedding_source.token_embedding.weight.detach()
    raise TypeError(
        "embedding_source must be either an embedding matrix tensor or a model with "
        "a token_embedding layer."
    )
