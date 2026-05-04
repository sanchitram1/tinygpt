"""
Proper instruction dataset that keeps each Prompt/Response pair intact.

Unlike TokenChunkDataset (which blindly chops into context_length windows),
this dataset preserves the full prompt and masks loss on non-response tokens.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class InstructionDataset(Dataset):
    """Each sample is a complete Prompt/Response pair, tokenized as one sequence.

    Labels are masked (-100) on prompt tokens so the model only learns to
    predict the response portion. If a pair exceeds context_length, the
    response is truncated from the end (the prompt is always kept intact).
    """

    def __init__(
        self,
        instruction_file: str,
        tokenizer: Tokenizer,
        context_length: int,
        delimiter: str = "<|endoftext|>",
        prompt_marker: str = "Prompt:",
        response_marker: str = "Response:",
        eos_token: str = "<eos>",
    ):
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.delimiter = delimiter
        self.prompt_marker = prompt_marker
        self.response_marker = response_marker
        self.eos_id = tokenizer.token_to_id(eos_token)
        if self.eos_id is None:
            raise ValueError(f"EOS token '{eos_token}' not in tokenizer vocab")

        self.samples: list[tuple[list[int], list[int]]] = []
        self._load(instruction_file)

    def _load(self, path: str) -> None:
        """Parse the instruction file into (prompt_ids, response_ids) pairs."""
        with open(path, encoding="utf-8") as f:
            raw = f.read()

        chunks = [c.strip() for c in raw.split(self.delimiter) if c.strip()]

        for chunk in chunks:
            prompt_ids, response_ids = self._parse_chunk(chunk)
            if prompt_ids and response_ids:
                self.samples.append((prompt_ids, response_ids))

        if not self.samples:
            raise ValueError(f"No valid instruction pairs found in {path}")

    def _parse_chunk(self, chunk: str) -> tuple[list[int], list[int]]:
        """Split a chunk into prompt and response token ID lists."""
        # Find the response marker boundary
        response_idx = chunk.find(f"\n{self.response_marker}")
        if response_idx == -1:
            response_idx = chunk.find(self.response_marker)
        if response_idx == -1:
            return [], []

        prompt_text = chunk[:response_idx].strip()
        # Skip past the response marker and the space after it
        response_start = response_idx + len(self.response_marker)
        if chunk[response_start:response_start + 1] == ":":
            response_start += 1
        response_text = chunk[response_start:].strip()

        prompt_ids = self.tokenizer.encode(prompt_text).ids
        response_ids = self.tokenizer.encode(response_text).ids + [self.eos_id]

        return prompt_ids, response_ids

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (input_ids, labels) where labels mask the prompt portion."""
        prompt_ids, response_ids = self.samples[idx]

        # Build the full sequence: prompt + response
        total_ids = prompt_ids + response_ids

        # Truncate if too long: keep prompt intact, cut response from end
        if len(total_ids) > self.context_length:
            prompt_len = len(prompt_ids)
            max_response_len = self.context_length - prompt_len
            if max_response_len <= 0:
                # Prompt alone exceeds context — keep last context_length tokens
                # but this is a degenerate case; skip by returning a shorter sample
                total_ids = total_ids[-self.context_length:]
                # Everything is "prompt" since response got cut entirely
                labels = [-100] * self.context_length
            else:
                total_ids = prompt_ids + response_ids[:max_response_len]
                labels = [-100] * prompt_len + response_ids[:max_response_len]
        else:
            labels = [-100] * len(prompt_ids) + response_ids

        # Pad to context_length
        pad_len = self.context_length - len(total_ids)
        input_ids = total_ids + [self.eos_id] * pad_len
        labels = labels + [-100] * pad_len

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )
