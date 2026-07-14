#!/usr/bin/env pkgx +python@3.13 uv run

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_suite.quality_metrics import bigram_repetition_ratio as _bigram_repetition_ratio
from eval_suite.quality_metrics import ends_cleanly as _ends_cleanly
from eval_suite.quality_metrics import entity_metrics as _entity_metrics
from eval_suite.quality_metrics import trigram_repetition_ratio as _trigram_repetition_ratio
from eval_suite.quality_metrics import weird_word_count as _weird_word_count
from eval_suite.quality_metrics import words as _words


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    texts = [row["generated_text"] for row in rows]
    return {
        "n": len(texts),
        "avg_words": round(statistics.mean(len(_words(text)) for text in texts), 1),
        "avg_chars": round(statistics.mean(len(text) for text in texts), 1),
        "morals": sum("moral" in text.lower() for text in texts),
        "unexpected": sum("unexpected" in text.lower() for text in texts),
        "quotes": sum('"' in text for text in texts),
        "clean_endings": sum(_ends_cleanly(text) for text in texts),
        "weird_words": sum(_weird_word_count(text) for text in texts),
        "avg_unique_names": round(
            statistics.mean(_entity_metrics(text)["unique_names"] for text in texts), 2
        ),
        "pronoun_mix_count": sum(
            _entity_metrics(text)["pronoun_mix"] for text in texts
        ),
        "adjacent_name_switches": sum(
            _entity_metrics(text)["adjacent_name_switches"] for text in texts
        ),
        "avg_entity_confusion_score": round(
            statistics.mean(
                _entity_metrics(text)["entity_confusion_score"] for text in texts
            ),
            2,
        ),
        "avg_bigram_repetition": round(
            statistics.mean(_bigram_repetition_ratio(text) for text in texts), 3
        ),
        "avg_trigram_repetition": round(
            statistics.mean(_trigram_repetition_ratio(text) for text in texts), 3
        ),
    }


def summarize_generations(generations: list[dict[str, Any]]) -> dict[str, Any]:
    flagged_generations = []
    for index, row in enumerate(generations):
        entity_metrics = _entity_metrics(row["generated_text"])
        if (
            entity_metrics["unique_names"] >= 4
            or entity_metrics["pronoun_mix"]
            or entity_metrics["adjacent_name_switches"] >= 2
        ):
            flagged_generations.append(
                {
                    "index": index,
                    "model": row.get("model"),
                    "prompt": row.get("prompt"),
                    "temperature": row.get("temperature"),
                    "top_k": row.get("top_k"),
                    **entity_metrics,
                }
            )

    by_model = {
        model: _summarize_rows(
            [row for row in generations if row.get("model") == model]
        )
        for model in sorted({row.get("model") for row in generations})
    }
    by_setting = {
        f"temperature={temperature}, top_k={top_k}": _summarize_rows(
            [
                row
                for row in generations
                if (row.get("temperature"), row.get("top_k")) == (temperature, top_k)
            ]
        )
        for temperature, top_k in sorted(
            {(row.get("temperature"), row.get("top_k")) for row in generations}
        )
    }
    return {
        "total_generations": len(generations),
        "models": dict(Counter(row.get("model") for row in generations)),
        "prompts": len({row.get("prompt") for row in generations}),
        "settings": {
            f"temperature={temperature}, top_k={top_k}": count
            for (temperature, top_k), count in Counter(
                (row.get("temperature"), row.get("top_k")) for row in generations
            ).items()
        },
        "by_model": by_model,
        "by_setting": by_setting,
        "flagged_generations": flagged_generations,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(f"Total generations: {summary['total_generations']}")
    print(f"Models: {summary['models']}")
    print(f"Prompt count: {summary['prompts']}")
    print(f"Settings: {summary['settings']}")
    print("\nBy model:")
    for model, metrics in summary["by_model"].items():
        print(f"  {model}: {metrics}")
    print("\nBy setting:")
    for setting, metrics in summary["by_setting"].items():
        print(f"  {setting}: {metrics}")
    print("\nFlagged entity/pronoun issues:")
    for row in summary["flagged_generations"][:10]:
        print(
            f"  index={row['index']} model={row['model']} "
            f"temp={row['temperature']} top_k={row['top_k']} "
            f"names={row['names']} pronoun_mix={row['pronoun_mix']} "
            f"name_switches={row['adjacent_name_switches']} "
            f"prompt={row['prompt']!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize generated story quality.")
    parser.add_argument("generations_path", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    generations = json.loads(args.generations_path.read_text(encoding="utf-8"))
    summary = summarize_generations(generations)
    print_summary(summary)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
