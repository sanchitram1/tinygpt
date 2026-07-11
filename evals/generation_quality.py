#!/usr/bin/env pkgx +python@3.13 uv run

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")
NAME_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
NON_NAME_WORDS = {
    "After",
    "As",
    "But",
    "Can",
    "Dad",
    "Every",
    "Everyone",
    "Finally",
    "Hello",
    "He",
    "Her",
    "His",
    "I",
    "Inside",
    "It",
    "Let",
    "Look",
    "Mom",
    "Mommy",
    "Mother",
    "Okay",
    "One",
    "Once",
    "She",
    "So",
    "Suddenly",
    "That",
    "Thank",
    "The",
    "There",
    "Then",
    "They",
    "This",
    "We",
    "What",
    "When",
    "Where",
    "Wow",
    "Yes",
    "You",
}
FEMALE_PRONOUNS = {"she", "her", "hers"}
MALE_PRONOUNS = {"he", "him", "his"}


def _words(text: str) -> list[str]:
    return [word.lower() for word in WORD_RE.findall(text)]


def _bigram_repetition_ratio(text: str) -> float:
    words = _words(text)
    if len(words) < 2:
        return 0.0
    bigrams = list(zip(words, words[1:]))
    return 1.0 - (len(set(bigrams)) / len(bigrams))


def _trigram_repetition_ratio(text: str) -> float:
    words = _words(text)
    if len(words) < 3:
        return 0.0
    trigrams = list(zip(words, words[1:], words[2:]))
    return 1.0 - (len(set(trigrams)) / len(trigrams))


def _weird_word_count(text: str) -> int:
    count = 0
    for word in _words(text):
        if "-" in word or len(word) >= 14:
            count += 1
        elif re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", word):
            count += 1
    return count


def _ends_cleanly(text: str) -> bool:
    return text.rstrip().endswith((".", "!", "?", '"'))


def _sentences(text: str) -> list[str]:
    return [
        sentence.strip() for sentence in SENTENCE_RE.findall(text) if sentence.strip()
    ]


def _names(text: str) -> list[str]:
    names = []
    for name in NAME_RE.findall(text):
        if name not in NON_NAME_WORDS:
            names.append(name)
    return names


def _entity_metrics(text: str) -> dict[str, Any]:
    names = _names(text)
    unique_names = sorted(set(names))
    words = _words(text)
    female_pronouns = sum(word in FEMALE_PRONOUNS for word in words)
    male_pronouns = sum(word in MALE_PRONOUNS for word in words)
    pronoun_mix = int(female_pronouns > 0 and male_pronouns > 0)

    adjacent_name_switches = 0
    previous_sentence_names: set[str] | None = None
    for sentence in _sentences(text):
        sentence_names = set(_names(sentence))
        if (
            sentence_names
            and previous_sentence_names
            and not (sentence_names & previous_sentence_names)
        ):
            adjacent_name_switches += 1
        if sentence_names:
            previous_sentence_names = sentence_names

    return {
        "unique_names": len(unique_names),
        "names": unique_names,
        "female_pronouns": female_pronouns,
        "male_pronouns": male_pronouns,
        "pronoun_mix": pronoun_mix,
        "adjacent_name_switches": adjacent_name_switches,
        "entity_confusion_score": len(unique_names)
        + pronoun_mix
        + adjacent_name_switches,
    }


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
