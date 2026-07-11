import argparse
import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import requests
from dotenv import load_dotenv

from config import DataConfig, TokenizationConfig
from tokenizer import iter_stories

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / "fine_tuning" / ".env"
PROVIDER_CONFIG = {
    "perplexity": {
        "api_url": "https://api.perplexity.ai/chat/completions",
        "api_key_env": "PERPLEXITY_API_KEY",
        "default_model": "sonar",
    },
    "openai": {
        "api_url": "https://api.openai.com/v1/chat/completions",
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4.1-mini",
    },
    "kimi": {
        "api_url": "https://api.moonshot.ai/v1/chat/completions",
        "api_key_env": "KIMI_API_KEY",
        "default_model": "kimi-k2-0905-preview",
    },
}


@dataclass(frozen=True)
class InstructionTokensConfig:
    prompt: str = "Prompt"
    response: str = "Response"


@dataclass(frozen=True)
class PromptSchema:
    length: str
    tone: str
    opening: str
    ending: str
    entity_focus: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample stories from TinyStories, assign controlled prompt attributes, "
            "and ask Perplexity to draft instruction prompts that match each story."
        ),
        epilog=(
            "Example:\n"
            "  uv run controlled-prompts --count 200 --split train "
            "--output-file data/fine_tuning-instructions-controlled.txt\n\n"
            "What gets controlled:\n"
            "  --lengths: story length requests like short / medium / long\n"
            "  --tones: tone requests like silly / gentle / spooky / bedtime\n"
            "  --openings: opening constraints such as a required starting phrase\n"
            "  --endings: ending constraints such as a cheerful or reflective ending\n"
            "  --entity-focuses: what the prompt should emphasize, like names, objects, or places"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of stories to sample and convert into Prompt/Response examples.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "valid"),
        default="train",
        help="Which original TinyStories split to sample from.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        help="Optional override for the source stories file. Defaults to the selected split.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=ROOT / "data" / "fine_tuning-instructions-controlled.txt",
        help="Path to the generated instruction dataset text file.",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=None,
        help="Optional JSON sidecar with sampled schema and raw API outputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for story sampling and schema assignment.",
    )
    parser.add_argument(
        "--provider",
        choices=tuple(PROVIDER_CONFIG),
        default="perplexity",
        help="Which API provider to use for prompt drafting.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name to use for prompt drafting. Defaults depend on --provider.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for provider prompt drafting.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=60,
        help="HTTP timeout for each Perplexity request.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.2,
        help="Delay between API calls to reduce burstiness.",
    )
    parser.add_argument(
        "--max-train-stories",
        type=int,
        default=1_000_000,
        help="Maximum number of original stories to scan while sampling.",
    )
    parser.add_argument(
        "--lengths",
        nargs="+",
        default=["short", "medium-length", "long"],
        help="Candidate length attributes to mix into prompts.",
    )
    parser.add_argument(
        "--tones",
        nargs="+",
        default=["gentle", "silly", "bedtime", "spooky", "playful"],
        help="Candidate tone attributes to mix into prompts.",
    )
    parser.add_argument(
        "--openings",
        nargs="+",
        default=[
            "no special opening",
            "start with 'Once upon a time'",
            "start by introducing the main character right away",
            "begin in the middle of a playful moment",
        ],
        help="Candidate opening constraints to mix into prompts.",
    )
    parser.add_argument(
        "--endings",
        nargs="+",
        default=[
            "no special ending",
            "end happily",
            "end with a gentle lesson",
            "end with everyone feeling proud",
            "end with a funny final image",
        ],
        help="Candidate ending constraints to mix into prompts.",
    )
    parser.add_argument(
        "--entity-focuses",
        nargs="+",
        default=[
            "focus on the main characters",
            "mention the most important object",
            "highlight the setting",
            "include the key relationship between the characters",
        ],
        help="Candidate entity-emphasis instructions to mix into prompts.",
    )
    return parser.parse_args()


def reservoir_sample(
    stories: Iterable[str], count: int, rng: random.Random
) -> list[str]:
    sample: list[str] = []
    for index, story in enumerate(stories):
        if index < count:
            sample.append(story)
            continue
        replacement_index = rng.randint(0, index)
        if replacement_index < count:
            sample[replacement_index] = story
    return sample


def source_file_for_split(args: argparse.Namespace, data_config: DataConfig) -> Path:
    if args.input_file:
        return args.input_file
    if args.split == "valid":
        return Path(data_config.validation_file_local)
    return Path(data_config.training_file_local)


WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
NAME_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")

# TODO: more elegant way pls
PLACE_HINTS = {
    "park",
    "house",
    "home",
    "school",
    "forest",
    "river",
    "shop",
    "store",
    "garden",
    "room",
    "yard",
    "road",
    "beach",
    "farm",
    "sky",
}
OBJECT_HINTS = {
    "ball",
    "bear",
    "box",
    "toy",
    "car",
    "kite",
    "book",
    "cake",
    "apple",
    "rock",
    "hat",
    "doll",
    "truck",
}
GENTLE_HINTS = {"kind", "happy", "gentle", "help", "friend", "hug", "smile", "love"}
SILLY_HINTS = {"funny", "laughed", "laugh", "silly", "giggle"}
SPOOKY_HINTS = {"dark", "scared", "afraid", "night", "monster", "shadow"}
BEDTIME_HINTS = {"sleep", "bed", "night", "dream", "blanket", "yawn"}


def _words(text: str) -> list[str]:
    return [word.lower() for word in WORD_RE.findall(text)]


def infer_length_label(story: str) -> str:
    word_count = len(_words(story))
    if word_count < 90:
        return "short"
    if word_count < 180:
        return "medium-length"
    return "long"


def infer_tone_label(story: str) -> str:
    words = set(_words(story))
    if words & SPOOKY_HINTS:
        return "spooky"
    if words & BEDTIME_HINTS:
        return "bedtime"
    if words & SILLY_HINTS:
        return "silly"
    if words & GENTLE_HINTS:
        return "gentle"
    return "playful"


def infer_opening_label(story: str) -> str:
    stripped = story.strip()
    lowered = stripped.lower()
    if lowered.startswith("once upon a time"):
        return "start with 'Once upon a time'"
    first_sentence = stripped.split(".", 1)[0].lower()
    if any(name.lower() in first_sentence for name in NAME_RE.findall(story)):
        return "start by introducing the main character right away"
    if any(place in first_sentence for place in PLACE_HINTS):
        return "begin by setting the scene"
    return "no special opening"


def infer_ending_label(story: str) -> str:
    stripped = story.strip()
    final_sentence = stripped.splitlines()[-1].strip().lower()
    if not final_sentence:
        return "no special ending"
    if "happily ever after" in final_sentence:
        return "end with 'happily ever after'"
    if "everyone ate cake" in final_sentence:
        return "end with 'everyone ate cake'"
    if any(
        phrase in final_sentence
        for phrase in ("happy", "best friends", "okay", "proud", "smile", "hug")
    ):
        return "end happily"
    if any(phrase in final_sentence for phrase in ("learned", "lesson", "remembered")):
        return "end with a gentle lesson"
    return "end with the problem being solved"


def infer_entity_focus(story: str) -> str:
    words = _words(story)
    word_set = set(words)
    names = NAME_RE.findall(story)
    place_count = sum(word in PLACE_HINTS for word in words)
    object_count = sum(word in OBJECT_HINTS for word in words)

    if len(names) >= 2:
        return "focus on the main characters"
    if place_count >= max(2, object_count):
        return "highlight the setting"
    if object_count >= 2:
        return "mention the most important object"
    if "friend" in word_set or "together" in word_set or "help" in word_set:
        return "include the key relationship between the characters"
    return "focus on the main characters"


def assign_schema(
    args: argparse.Namespace, rng: random.Random, story: str
) -> PromptSchema:
    inferred_length = infer_length_label(story)
    inferred_tone = infer_tone_label(story)
    inferred_opening = infer_opening_label(story)
    inferred_ending = infer_ending_label(story)
    inferred_entity_focus = infer_entity_focus(story)

    allowed_lengths = [value for value in args.lengths if value == inferred_length]
    allowed_tones = [value for value in args.tones if value == inferred_tone]
    allowed_openings = [value for value in args.openings if value == inferred_opening]
    allowed_endings = [value for value in args.endings if value == inferred_ending]
    allowed_entity_focuses = [
        value for value in args.entity_focuses if value == inferred_entity_focus
    ]

    return PromptSchema(
        length=(rng.choice(allowed_lengths) if allowed_lengths else inferred_length),
        tone=(rng.choice(allowed_tones) if allowed_tones else inferred_tone),
        opening=(
            rng.choice(allowed_openings) if allowed_openings else inferred_opening
        ),
        ending=(rng.choice(allowed_endings) if allowed_endings else inferred_ending),
        entity_focus=(
            rng.choice(allowed_entity_focuses)
            if allowed_entity_focuses
            else inferred_entity_focus
        ),
    )


def build_system_prompt() -> str:
    return (
        "You are creating supervised fine-tuning prompts for a TinyStories-style model. "
        "Given a children's story and a prompt schema, write exactly one user instruction "
        "that would naturally ask for that story while honoring the schema.\n\n"
        "Requirements:\n"
        "- Output only the final prompt text.\n"
        "- Keep it concise, natural, and specific.\n"
        "- Make the prompt sound like something a user would type.\n"
        "- Reflect the requested length, tone, opening, ending, and entity focus when possible.\n"
        "- Do not request an exact opening or ending phrase unless it is already part of the story.\n"
        "- Prefer softer descriptions like happy ending, gentle lesson, or scene-setting opener when exact phrasing is not present.\n"
        "- Do not include labels, markdown, bullet points, or explanation.\n"
        "- Do not mention the schema explicitly.\n"
    )


def build_user_message(story: str, schema: PromptSchema) -> str:
    return (
        "Draft one instruction prompt for the following story.\n\n"
        f"Schema:\n{json.dumps(asdict(schema), indent=2)}\n\n"
        f"Story:\n{story}"
    )


def generate_prompt_for_story(
    provider: str,
    api_key: str,
    model_name: str,
    story: str,
    schema: PromptSchema,
    temperature: float,
    timeout_seconds: int,
) -> str:
    provider_settings = PROVIDER_CONFIG[provider]
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_user_message(story, schema)},
        ],
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(
        provider_settings["api_url"],
        headers=headers,
        data=json.dumps(payload),
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"].strip()
    return content.splitlines()[0].strip().strip('"')


def flush_outputs(
    output_file: Path,
    metadata_file: Path,
    generated_examples: list[str],
    metadata_rows: list[dict],
    delimiter: str,
) -> None:
    output_text = f"\n{delimiter}\n\n".join(generated_examples)
    if output_text:
        output_text += f"\n{delimiter}\n"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(output_text, encoding="utf-8")
    metadata_file.write_text(json.dumps(metadata_rows, indent=2), encoding="utf-8")


def load_existing_progress(
    output_file: Path,
    metadata_file: Path,
    delimiter: str,
) -> tuple[list[str], list[dict]]:
    generated_examples: list[str] = []
    metadata_rows: list[dict] = []

    if output_file.exists():
        raw_text = output_file.read_text(encoding="utf-8")
        generated_examples = [
            chunk.strip() for chunk in raw_text.split(delimiter) if chunk.strip()
        ]

    if metadata_file.exists():
        metadata_rows = json.loads(metadata_file.read_text(encoding="utf-8"))

    if (
        generated_examples
        and metadata_rows
        and len(generated_examples) != len(metadata_rows)
    ):
        raise ValueError(
            "Existing output and metadata files disagree on completed example count."
        )

    return generated_examples, metadata_rows


def main() -> None:
    args = parse_args()
    load_dotenv(dotenv_path=ENV_PATH)

    provider_settings = PROVIDER_CONFIG[args.provider]
    api_key = os.environ.get(provider_settings["api_key_env"])
    if not api_key:
        raise RuntimeError(
            "Missing API key for provider "
            f"{args.provider!r}. Expected {provider_settings['api_key_env']} in the "
            f"environment or {ENV_PATH}."
        )
    model_name = args.model or provider_settings["default_model"]

    rng = random.Random(args.seed)
    tokenization_config = TokenizationConfig(max_train_stories=args.max_train_stories)
    instruction_tokens = InstructionTokensConfig()
    data_config = DataConfig()

    source_file = source_file_for_split(args, data_config)
    sampled_stories = reservoir_sample(
        iter_stories(tokenization_config, source_file),
        count=args.count,
        rng=rng,
    )

    metadata_path = args.metadata_file
    if metadata_path is None:
        metadata_path = args.output_file.with_suffix(".metadata.json")

    delimiter = tokenization_config.story_delimiter
    generated_examples, metadata_rows = load_existing_progress(
        args.output_file,
        metadata_path,
        delimiter,
    )
    completed = len(generated_examples)
    if completed:
        print(f"Resuming from existing progress: {completed} completed example(s)")

    for index, story in enumerate(sampled_stories, start=1):
        if index <= completed:
            continue
        schema = assign_schema(args, rng, story)
        prompt = generate_prompt_for_story(
            provider=args.provider,
            api_key=api_key,
            model_name=model_name,
            story=story,
            schema=schema,
            temperature=args.temperature,
            timeout_seconds=args.timeout_seconds,
        )
        generated_examples.append(
            f"{instruction_tokens.prompt}: {prompt}\n"
            f"{instruction_tokens.response}: {story}"
        )
        metadata_rows.append(
            {
                "index": index,
                "provider": args.provider,
                "model": model_name,
                "prompt": prompt,
                "schema": asdict(schema),
                "story_preview": story[:200],
            }
        )
        flush_outputs(
            args.output_file,
            metadata_path,
            generated_examples,
            metadata_rows,
            delimiter,
        )
        print(f"[{index}/{len(sampled_stories)}] drafted prompt")
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    flush_outputs(
        args.output_file,
        metadata_path,
        generated_examples,
        metadata_rows,
        delimiter,
    )
    print(f"Wrote {len(generated_examples)} examples to {args.output_file}")
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
