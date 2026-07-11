#!/usr/bin/env pkgx +python@3.13 uv run
"""
Cleanse template-generated prompts by detecting fake character names
and using OpenAI to regenerate proper prompts for those examples.

Detection: extracts character-like words from the prompt, checks if each
one is a real name in the story (appears 2+ times or mid-sentence).
If any are fake, the example is flagged for repair.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from dotenv import load_dotenv

from config import TokenizationConfig
from tokenizer import iter_stories

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / "fine_tuning" / ".env"

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
NAME_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")

# Same NOT_NAMES as template_prompts.py
NOT_NAMES = {
    "once", "one", "day", "after", "then", "but", "and", "the", "a", "an",
    "he", "she", "they", "his", "her", "him", "it", "its", "we", "you",
    "this", "that", "these", "those", "there", "here", "where", "when", "what",
    "who", "why", "how", "if", "so", "or", "as", "at", "by", "in", "on",
    "to", "from", "for", "with", "without", "about", "into", "onto", "upon",
    "all", "every", "each", "some", "any", "no", "not", "now",
    "just", "only", "also", "even", "still", "again", "too", "very",
    "suddenly", "finally", "soon", "later", "first", "next", "last",
    "maybe", "perhaps", "however", "because", "since", "although", "while",
    "let", "lets", "look", "see", "come", "go", "get", "make", "put", "take",
    "think", "know", "want", "need", "like", "love", "say", "said", "tell",
    "ask", "give", "find", "help", "try", "use", "keep", "start", "stop",
    "begin", "end", "turn", "walk", "run", "jump", "play", "eat", "drink",
    "sleep", "wake", "sit", "stand", "open", "close", "pick", "choose",
    "everyone", "everything", "everybody", "something", "anything",
    "nothing", "someone", "anyone", "nobody", "hello", "goodbye", "please",
    "thank", "sorry", "okay", "yes", "yeah", "wow", "oh", "ah",
    "moral", "lesson", "story",
    "mom", "mommy", "dad", "daddy", "mother", "father", "sister", "brother",
    "grandma", "grandpa", "grandmother", "grandfather", "uncle", "aunt",
    "cousin", "baby",
    "mr", "mrs", "ms", "miss", "sir", "dr",
    "short", "medium", "long", "keep", "make",
    # Template keywords that appear capitalized in prompts
    "write", "tell", "create", "make", "give", "feature",
    "theme", "tone", "ending", "style", "length", "focus",
    "prompt", "response", "ends", "end",
    # The word "I" and contractions
    "i", "im", "ive", "id", "ill",
    # More false-name words from templates
    "ends", "all", "just", "good", "everyone", "inside",
    "always", "never", "let", "from", "hello", "suddenly",
}


def _is_real_name(candidate: str, story: str) -> bool:
    lowered = candidate.lower()
    if lowered in NOT_NAMES:
        return False
    if len(candidate) < 3:
        return False
    count = len(re.findall(rf"\b{re.escape(candidate)}\b", story))
    if count >= 2:
        return True
    mid = bool(re.search(rf"[^.!?\n\s]\s+{re.escape(candidate)}\b", story))
    return mid


def extract_prompt_names(prompt_text: str) -> list[str]:
    """Extract words from the prompt that look like character references."""
    # Remove the "Prompt: " prefix
    text = prompt_text.replace("Prompt:", "").strip()
    # Find capitalized words that aren't in NOT_NAMES
    candidates = NAME_RE.findall(text)
    return [c for c in candidates if c.lower() not in NOT_NAMES and len(c) >= 3]


def is_bad_example(prompt: str, story: str) -> tuple[bool, list[str]]:
    """Check if the prompt references fake character names not in the story."""
    names = extract_prompt_names(prompt)
    fake = [n for n in names if not _is_real_name(n, story)]
    return len(fake) > 0, fake


def parse_examples(file_path: Path, delimiter: str) -> list[dict]:
    """Parse a Prompt/Response file into a list of dicts."""
    raw = file_path.read_text(encoding="utf-8")
    chunks = [c.strip() for c in raw.split(delimiter) if c.strip()]
    examples = []
    for chunk in chunks:
        if "Response:" not in chunk:
            continue
        # Split into prompt and response
        resp_idx = chunk.find("\nResponse:")
        if resp_idx == -1:
            resp_idx = chunk.find("Response:")
        if resp_idx == -1:
            continue
        prompt = chunk[:resp_idx].strip()
        response_marker_end = resp_idx + len("Response:")
        if chunk[response_marker_end:response_marker_end + 1] in (" ", "\n"):
            pass
        response = chunk[response_marker_end:].strip()
        examples.append({"prompt": prompt, "response": response, "raw": chunk})
    return examples


SYSTEM_PROMPT = """You are fixing bad prompts in a children's story fine-tuning dataset.

Given a children's story, the original prompt mentioned fake character names that don't exist in the story. Your job: write a CORRECT prompt that accurately describes the story.

Rules:
- Output ONLY the fixed prompt text. No explanation, no quotes, no markdown.
- The prompt should sound like a natural user request.
- Mention the real characters, tone, and theme from the story.
- Keep it under 25 words.
- If the story is about animals, mention the animal types.
- If the story has named characters, use their actual names. If no named characters exist, refer to "the main character" or describe the situation instead."""


def fix_prompt_via_openai(api_key: str, story: str, fake_names: list[str]) -> str:
    """Ask OpenAI to generate a corrected prompt for the story."""
    user_msg = (
        f"The original prompt had these fake names: {', '.join(fake_names)}.\n\n"
        f"Story:\n{story}\n\n"
        f"Write a corrected prompt for this story."
    )
    payload = {
        "model": "gpt-4.1-mini",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.3,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        data=json.dumps(payload),
        timeout=60,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"].strip()
    return content.splitlines()[0].strip().strip('"')


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect and fix fake-name prompts in template-generated data."
    )
    parser.add_argument(
        "input_file", type=Path,
        help="Template-generated instruction file to cleanse.",
    )
    parser.add_argument(
        "--output-file", type=Path,
        help="Output path for cleansed file.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only detect bad examples, don't fix them.",
    )
    parser.add_argument(
        "--max-fix", type=int, default=500,
        help="Maximum number of examples to fix via API.",
    )
    parser.add_argument(
        "--sleep-seconds", type=float, default=0.1,
    )
    args = parser.parse_args()

    load_dotenv(dotenv_path=ENV_PATH)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.dry_run:
        raise RuntimeError(f"Missing OPENAI_API_KEY in {ENV_PATH}")

    delimiter = "<|endoftext|>"
    examples = parse_examples(args.input_file, delimiter)
    print(f"Loaded {len(examples)} examples from {args.input_file}")

    # Detect bad examples
    bad_indices = []
    for i, ex in enumerate(examples):
        is_bad, fake_names = is_bad_example(ex["prompt"], ex["response"])
        if is_bad:
            bad_indices.append(i)
            if len(bad_indices) <= 20:  # Print first 20
                print(f"  BAD [{i}]: fake={fake_names}  prompt={ex['prompt'][:100]}...")

    pct = 100 * len(bad_indices) / max(1, len(examples))
    print(f"\nFound {len(bad_indices)} bad examples ({pct:.1f}%)")

    if args.dry_run:
        print("Dry run — no fixes applied.")
        return

    # Fix bad examples
    to_fix = bad_indices[:args.max_fix]
    print(f"Fixing {len(to_fix)} examples via OpenAI...")

    fixed_count = 0
    for idx in to_fix:
        ex = examples[idx]
        _, fake_names = is_bad_example(ex["prompt"], ex["response"])
        try:
            new_prompt = fix_prompt_via_openai(api_key, ex["response"], fake_names)
            examples[idx]["prompt"] = f"Prompt: {new_prompt}"
            fixed_count += 1
            if fixed_count % 20 == 0:
                print(f"  Fixed {fixed_count}/{len(to_fix)}")
        except Exception as e:
            print(f"  ERROR [{idx}]: {e}", file=sys.stderr)
        time.sleep(args.sleep_seconds)

    print(f"Fixed {fixed_count} examples")

    # Rebuild output
    output_path = args.output_file or args.input_file.with_suffix(".cleansed.txt")
    rebuilt = []
    for ex in examples:
        rebuilt.append(f"{ex['prompt']}\nResponse: {ex['response']}")

    output_text = ("\n" + delimiter + "\n\n").join(rebuilt) + "\n" + delimiter + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output_text, encoding="utf-8")
    print(f"Wrote {len(rebuilt)} examples to {output_path}")


if __name__ == "__main__":
    main()
