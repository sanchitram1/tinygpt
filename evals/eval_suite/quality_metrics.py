"""Import-safe text quality metrics shared across evals tooling.

These functions were extracted from ``evals/generation_quality.py`` so they can
be reused by the evaluation runner without depending on that script's CLI. The
CLI module imports from here; behavior is unchanged.
"""

from __future__ import annotations

import re
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


def words(text: str) -> list[str]:
    return [word.lower() for word in WORD_RE.findall(text)]


def bigram_repetition_ratio(text: str) -> float:
    tokens = words(text)
    if len(tokens) < 2:
        return 0.0
    bigrams = list(zip(tokens, tokens[1:]))
    return 1.0 - (len(set(bigrams)) / len(bigrams))


def trigram_repetition_ratio(text: str) -> float:
    tokens = words(text)
    if len(tokens) < 3:
        return 0.0
    trigrams = list(zip(tokens, tokens[1:], tokens[2:]))
    return 1.0 - (len(set(trigrams)) / len(trigrams))


def weird_word_count(text: str) -> int:
    count = 0
    for word in words(text):
        if "-" in word or len(word) >= 14:
            count += 1
        elif re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", word):
            count += 1
    return count


def ends_cleanly(text: str) -> bool:
    return text.rstrip().endswith((".", "!", "?", '"'))


def sentences(text: str) -> list[str]:
    return [
        sentence.strip() for sentence in SENTENCE_RE.findall(text) if sentence.strip()
    ]


def names(text: str) -> list[str]:
    found = []
    for name in NAME_RE.findall(text):
        if name not in NON_NAME_WORDS:
            found.append(name)
    return found


def entity_metrics(text: str) -> dict[str, Any]:
    found_names = names(text)
    unique_names = sorted(set(found_names))
    tokens = words(text)
    female_pronouns = sum(word in FEMALE_PRONOUNS for word in tokens)
    male_pronouns = sum(word in MALE_PRONOUNS for word in tokens)
    pronoun_mix = int(female_pronouns > 0 and male_pronouns > 0)

    adjacent_name_switches = 0
    previous_sentence_names: set[str] | None = None
    for sentence in sentences(text):
        sentence_names = set(names(sentence))
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
