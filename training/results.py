from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class TrainingResult:
    config: dict[str, Any]
    parameter_count: int
    train_history: list[dict[str, Any]]
    valid_history: list[dict[str, Any]]
    model_path: str
    total_training_time_seconds: float
    tokens_processed: int
    tokens_per_second: float

    @property
    def validation_losses(self) -> list[float]:
        return [float(entry["loss"]) for entry in self.valid_history]

    @property
    def validation_steps(self) -> list[int]:
        return [int(entry["step"]) for entry in self.valid_history]

    @property
    def training_losses(self) -> list[float]:
        return [float(entry["loss"]) for entry in self.train_history]

    @property
    def training_steps(self) -> list[int]:
        return [int(entry["step"]) for entry in self.train_history]


def load_training_result(metrics_path: str | Path) -> TrainingResult:
    metrics_path = Path(metrics_path)
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    return TrainingResult(
        config=payload["config"],
        parameter_count=int(payload["parameter_count"]),
        train_history=list(payload["train_history"]),
        valid_history=list(payload["valid_history"]),
        model_path=str(payload["model_path"]),
        total_training_time_seconds=float(payload["total_training_time_seconds"]),
        tokens_processed=int(payload["tokens_processed"]),
        tokens_per_second=float(payload["tokens_per_second"]),
    )


def load_results_by_name(
    metrics_paths: dict[str, str | Path],
) -> dict[str, TrainingResult]:
    return {
        model_name: load_training_result(metrics_path)
        for model_name, metrics_path in metrics_paths.items()
    }


def load_generations(generations_path: str | Path) -> list[dict[str, Any]]:
    generations_path = Path(generations_path)
    return list(json.loads(generations_path.read_text(encoding="utf-8")))


def generations_to_dataframe(
    generations: list[dict[str, Any]],
    temperature: float | None = None,
    top_k: int | None = None,
    value_column: str = "generated_text",
) -> pd.DataFrame:
    filtered = generations
    if temperature is not None:
        filtered = [
            row
            for row in filtered
            if float(row.get("temperature")) == float(temperature)
        ]
    if top_k is not None:
        filtered = [row for row in filtered if int(row.get("top_k")) == int(top_k)]

    frame = pd.DataFrame(filtered)
    if frame.empty:
        return pd.DataFrame()

    return frame.pivot_table(
        index="prompt",
        columns="model",
        values=value_column,
        aggfunc="first",
    ).sort_index()
