#!/usr/bin/env pkgx +python@3.13 uv run
"""
Template-based instruction data generator for TinyGPT fine-tuning.

Analyzes stories to infer their attributes (length, tone, opening, ending, theme),
then generates prompt/response pairs using natural-language templates.

No API calls — processes thousands of stories in seconds.
"""

from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from config import TokenizationConfig
from tokenizer import iter_stories

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
NAME_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")

# --- Inference helpers (mirror controlled_prompts.py) ---

GENTLE_HINTS = {"kind", "happy", "gentle", "help", "friend", "hug", "smile", "love"}
SILLY_HINTS = {"funny", "laughed", "laugh", "silly", "giggle"}
SPOOKY_HINTS = {"dark", "scared", "afraid", "night", "monster", "shadow"}
BEDTIME_HINTS = {"sleep", "bed", "night", "dream", "blanket", "yawn"}
SAD_HINTS = {"sad", "cried", "cry", "lonely", "alone", "tears", "upset", "missed"}

ANIMAL_WORDS = {
    "cat", "dog", "bird", "fish", "rabbit", "bear", "fox", "mouse", "frog",
    "duck", "pig", "cow", "horse", "sheep", "lion", "tiger", "elephant",
    "monkey", "turtle", "owl", "hen", "chicken", "wolf", "deer", "squirrel",
    "puppy", "kitten", "bunny", "cub", "crocodile", "dinosaur", "dragon",
}
FRIENDSHIP_HINTS = {"friend", "together", "share", "shared", "help", "helped", "best friend"}
FAMILY_HINTS = {"mom", "dad", "mother", "father", "sister", "brother", "grandma", "grandpa", "uncle", "aunt", "family"}
SHARING_HINTS = {"share", "shared", "gave", "give", "offered", "half"}
ADVENTURE_HINTS = {"adventure", "explore", "journey", "discovered", "found", "went to", "climbed"}


def _words(text: str) -> list[str]:
    return [word.lower() for word in WORD_RE.findall(text)]


def infer_length_label(story: str) -> str:
    wc = len(_words(story))
    if wc < 80:
        return "short"
    if wc < 160:
        return "medium-length"
    return "long"


def infer_tone_label(story: str) -> str:
    words = set(_words(story))
    if words & SPOOKY_HINTS:
        return "spooky"
    if words & SAD_HINTS:
        return "sad"
    if words & BEDTIME_HINTS:
        return "cozy bedtime"
    if words & SILLY_HINTS:
        return "silly and funny"
    if words & GENTLE_HINTS:
        return "gentle and sweet"
    return "playful"


def infer_ending_label(story: str) -> str:
    stripped = story.strip()
    final = stripped.splitlines()[-1].strip().lower() if stripped else ""
    if "happily ever after" in final:
        return "with a happily-ever-after ending"
    if any(w in final for w in ("happy", "smile", "hug", "best friends", "proud")):
        return "with a happy ending"
    if any(w in final for w in ("learned", "lesson", "remembered", "never again")):
        return "with a gentle lesson at the end"
    if any(w in final for w in ("solved", "fixed", "okay", "alright")):
        return "where the problem gets solved"
    return "with a satisfying ending"


def infer_theme(story: str) -> str:
    words = set(_words(story))
    lowered = story.lower()
    if words & ANIMAL_WORDS:
        return "animals"
    if words & SHARING_HINTS:
        return "sharing and kindness"
    if words & FRIENDSHIP_HINTS:
        return "friendship"
    if words & FAMILY_HINTS:
        return "family"
    if words & ADVENTURE_HINTS:
        return "adventure and discovery"
    return "everyday life"


# Expanded list: sentence-starters and common capitalized function words
# that are NOT character names in children's stories
NOT_NAMES = {
    # Sentence starters / function words (lowercase for matching)
    "once", "one", "day", "after", "then", "but", "and", "the", "a", "an",
    "he", "she", "they", "his", "her", "him", "it", "its", "we", "you",
    "this", "that", "these", "those", "there", "here", "where", "when", "what",
    "who", "why", "how", "if", "so", "or", "as", "at", "by", "in", "on",
    "to", "from", "for", "with", "without", "about", "into", "onto", "upon",
    "all", "every", "each", "some", "any", "no", "not", "now", "then",
    "just", "only", "also", "even", "still", "again", "too", "very",
    "suddenly", "finally", "soon", "later", "first", "next", "last",
    "maybe", "perhaps", "however", "because", "since", "although", "while",
    "let", "lets", "look", "see", "come", "go", "get", "make", "put", "take",
    "think", "know", "want", "need", "like", "love", "say", "said", "tell",
    "ask", "give", "find", "help", "try", "use", "keep", "start", "stop",
    "begin", "end", "turn", "walk", "run", "jump", "play", "eat", "drink",
    "sleep", "wake", "sit", "stand", "open", "close", "pick", "choose",
    "every", "everyone", "everything", "everybody", "something", "anything",
    "nothing", "someone", "anyone", "nobody", "hello", "goodbye", "please",
    "thank", "sorry", "okay", "yes", "yeah", "wow", "oh", "ah",
    "moral", "lesson", "story",
    # Family (lowercase)
    "mom", "mommy", "dad", "daddy", "mother", "father", "sister", "brother",
    "grandma", "grandpa", "grandmother", "grandfather", "uncle", "aunt",
    "cousin", "baby",
    # Never names
    "mr", "mrs", "ms", "miss", "sir", "dr",
}

def _is_real_name(candidate: str, story: str) -> bool:
    """A capitalized word is a real name if it appears mid-sentence or 2+ times."""
    lowered = candidate.lower()
    if lowered in NOT_NAMES:
        return False
    # Must be at least 3 chars
    if len(candidate) < 3:
        return False
    # Count occurrences (case-sensitive for names)
    count = len(re.findall(rf"\b{re.escape(candidate)}\b", story))
    if count >= 2:
        return True
    # Single occurrence: only valid if NOT at sentence start
    # Check if it appears mid-sentence (preceded by non-period, non-start)
    mid_sentence = bool(re.search(rf"[^.!?\n\s]\s+{re.escape(candidate)}\b", story))
    return mid_sentence


def infer_characters(story: str) -> str:
    words = set(_words(story))
    animals = words & ANIMAL_WORDS
    if animals:
        animal_list = sorted(animals)[:3]
        if len(animal_list) == 1:
            return f"a {animal_list[0]}"
        return f"{', '.join(animal_list[:-1])} and {animal_list[-1]}"

    # Find real names: capitalized words that pass the _is_real_name check
    candidates = NAME_RE.findall(story)
    real_names = sorted(set(n for n in candidates if _is_real_name(n, story)))
    if real_names:
        name_list = real_names[:2]
        if len(name_list) == 1:
            return name_list[0]
        return f"{name_list[0]} and {name_list[1]}"

    return "the main character"


# --- Template system ---

@dataclass
class StoryAttrs:
    length: str
    tone: str
    ending: str
    theme: str
    characters: str


def extract_attrs(story: str) -> StoryAttrs:
    return StoryAttrs(
        length=infer_length_label(story),
        tone=infer_tone_label(story),
        ending=infer_ending_label(story),
        theme=infer_characters(story),
        characters=infer_characters(story),
    )


# Templates that combine attributes in natural ways
TEMPLATES = [
    # Length + tone + character focus
    "Write a {length} story about {characters} with a {tone} tone.",
    "Tell me a {length}, {tone} story about {characters}.",
    # Tone + ending + lesson
    "Write a {tone} story for kids that ends {ending} and features {characters}.",
    "I want a {length} children's story with a {tone} feeling, {ending}.",
    # Character-driven
    "Make up a {length} tale about {characters}. Make it {tone} and end it {ending}.",
    # Theme + emotion
    "Tell a {tone} story about {characters}. Keep it {length}.",
    # Length-oriented
    "Give me a {length} bedtime story about {characters} that feels {tone}.",
    # Ending-oriented
    "Write a children's story {ending}, featuring {characters}. Keep the tone {tone} and length {length}.",
    # Simple + direct
    "Tell me about {characters} in a {tone} story, {ending}.",
    # Complex request
    "Write a {length} story where {characters} learn something important. Make it {tone} and end {ending}.",
    # Emotion-focused
    "Write a {tone} children's story about {characters} that is {length}.",
    # Classic style
    "I'd like a {length}, {tone} fairy tale about {characters} {ending}.",
    # Friend-focused
    "Create a {tone} story about friends. Feature {characters}, make it {length}, {ending}.",
    # Adventure style
    "Tell me a {tone} adventure story about {characters}, {length}, {ending}.",
    # Simple bedtime
    "Write a {tone} story, {length} long, about {characters} {ending}.",
]

# Extra: some templates have optional fields to increase diversity
EXTRA_TEMPLATES = [
    "Write a {length} story. Feature {characters}. Make it {tone}.",
    "A {tone} tale about {characters}. {length}. Ends {ending}.",
    "I need a {length} children's story. Theme: {characters}. Tone: {tone}. Ending: {ending}.",
    "Create a story: {characters}, {tone}, {length}, {ending}.",
    "Please write about {characters}. Style: {tone}. Length: {length}. {ending}.",
]


def fill_template(template: str, attrs: StoryAttrs, rng: random.Random) -> str:
    """Fill a template, randomly varying some attributes for diversity."""
    # Occasionally flip the length for variation (makes the model learn the concept)
    length = attrs.length
    ending = attrs.ending
    tone = attrs.tone

    # 10% chance to use a different length (helps model learn length distinction)
    if rng.random() < 0.10:
        others = [l for l in ("short", "medium-length", "long") if l != length]
        length = rng.choice(others)

    # 10% chance to vary the ending description
    if rng.random() < 0.10:
        endings = [
            "with a happy ending",
            "with a gentle lesson at the end",
            "where the problem gets solved",
            "with a satisfying ending",
            "with a happily-ever-after ending",
        ]
        ending = rng.choice(endings)

    return template.format(
        length=length,
        tone=tone,
        ending=ending,
        characters=attrs.characters,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate template-based instruction data for TinyGPT fine-tuning."
    )
    parser.add_argument(
        "--count", type=int, default=2_000,
        help="Number of stories to sample.",
    )
    parser.add_argument(
        "--templates-per-story", type=int, default=3,
        help="How many prompt variants per story.",
    )
    parser.add_argument(
        "--split", choices=("train", "valid"), default="train",
        help="Which TinyStories split to use.",
    )
    parser.add_argument(
        "--output-file", type=Path,
        default=Path("data/fine_tuning-instructions-templates.txt"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-stories", type=int, default=1_000_000)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    tokenization_config = TokenizationConfig(
        max_train_stories=args.max_train_stories
    )

    # Find source file
    from config import DataConfig
    data_config = DataConfig()
    source_file = (
        Path(data_config.training_file_local) if args.split == "train"
        else Path(data_config.validation_file_local)
    )

    # Reservoir sample stories
    import sys
    sample: list[str] = []
    for i, story in enumerate(iter_stories(tokenization_config, source_file)):
        if i < args.count:
            sample.append(story)
        else:
            j = rng.randint(0, i)
            if j < args.count:
                sample[j] = story
        if (i + 1) % 10000 == 0:
            sys.stderr.write(f"\r  Scanned {i+1:,} stories...")
            sys.stderr.flush()

    print(f"\r  Sampled {len(sample)} stories from {i+1:,} total.         ")

    # Generate prompt/response pairs
    all_templates = TEMPLATES + EXTRA_TEMPLATES
    examples: list[str] = []
    for story in sample:
        attrs = extract_attrs(story)
        chosen = rng.sample(all_templates, min(args.templates_per_story, len(all_templates)))
        for tmpl in chosen:
            prompt = fill_template(tmpl, attrs, rng)
            examples.append(f"Prompt: {prompt}\nResponse: {story}")

    # Shuffle to mix templates
    rng.shuffle(examples)

    # Write output
    delimiter = tokenization_config.story_delimiter
    output_text = f"\n{delimiter}\n\n".join(examples) + f"\n{delimiter}\n"
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(output_text, encoding="utf-8")

    print(f"Wrote {len(examples)} examples ({args.templates_per_story} templates × {len(sample)} stories) to {args.output_file}")
    print(f"File size: {args.output_file.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
