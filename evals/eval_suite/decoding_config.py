"""Versioned decoding configuration for the evaluation suite.

Locks temperature, top_k, max_new_tokens, and the global suite seed so every
candidate model is run under identical decoding settings. See
``evals/configs/README.md``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema

DECODING_CONFIG_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "decoding-config",
    "type": "object",
    "required": ["version", "temperature", "top_k", "max_new_tokens", "suite_seed"],
    "additionalProperties": False,
    "properties": {
        "version": {"type": "string", "pattern": r"^v\d+\.\d+$"},
        "temperature": {"type": "number", "exclusiveMinimum": 0},
        "top_k": {"type": "integer", "minimum": 1},
        "max_new_tokens": {"type": "integer", "minimum": 1},
        "suite_seed": {"type": "integer"},
    },
}


class DecodingConfigError(ValueError):
    pass


@dataclass(frozen=True)
class DecodingConfig:
    version: str
    temperature: float
    top_k: int
    max_new_tokens: int
    suite_seed: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "max_new_tokens": self.max_new_tokens,
            "suite_seed": self.suite_seed,
        }

    def case_seed(self, case_id: str) -> int:
        """Deterministically derive a per-case generation seed.

        Combines the suite seed and case id so the seed is stable across runs
        and machines, and recorded per case as required by the eval protocol,
        without needing a seed column hand-maintained in the dataset file.
        """
        digest = hashlib.sha256(f"{self.suite_seed}:{case_id}".encode("utf-8")).hexdigest()
        return int(digest[:16], 16) % (2**32)


def validate_decoding_config(record: dict[str, Any]) -> None:
    try:
        jsonschema.validate(instance=record, schema=DECODING_CONFIG_SCHEMA)
    except jsonschema.ValidationError as exc:
        raise DecodingConfigError(f"invalid decoding config: {exc.message}") from exc


def load_decoding_config(path: Path) -> DecodingConfig:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_decoding_config(record)
    return DecodingConfig(**record)
