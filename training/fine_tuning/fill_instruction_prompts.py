#!/usr/bin/env pkgx uv run
import json
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from dotenv import load_dotenv

from config import TokenizationConfig


@dataclass(frozen=True)
class InstructionTokensConfig:
    prompt: str = "<prompt>"
    response: str = "<response>"


ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / "data" / "fine_tuning-instructions-valid_empty.txt"
OUT_PATH = ROOT / "data" / "fine_tuning-instructions-valid.txt"
FINE_TUNING_README_PATH = ROOT / "fine_tuning" / "README.md"
ENV_PATH = ROOT / "fine_tuning" / ".env"

PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
PERPLEXITY_MODEL = "sonar"


def parse_examples(
    raw_text: str, delimiter: str, prompt_tag: str, response_tag: str
) -> list[str]:
    chunks = [chunk.strip() for chunk in raw_text.split(delimiter) if chunk.strip()]
    return [normalize_chunk(chunk, prompt_tag, response_tag) for chunk in chunks]


def normalize_chunk(chunk: str, prompt_tag: str, response_tag: str) -> str:
    prompt_prefix = f"{prompt_tag}:"
    response_prefix = f"{response_tag}:"

    if prompt_prefix not in chunk or response_prefix not in chunk:
        raise ValueError("Chunk missing prompt/response tags.")

    response_idx = chunk.index(response_prefix)
    response_body = chunk[response_idx + len(response_prefix) :].strip()
    return response_body


def build_system_prompt(readme_context: str) -> str:
    return (
        "You are helping build supervised fine-tuning data for a TinyStories model. "
        "Given a children's story, generate a single concise user-style prompt that "
        "would naturally ask for that exact story. Keep it age-appropriate, specific "
        "to the story content, and under 20 words. "
        "Do not include labels, quotation marks, extra explanation, or markdown.\n\n"
        "Project context:\n"
        f"{readme_context}"
    )


def generate_prompt_for_response(
    api_key: str, system_prompt: str, response_text: str, timeout_seconds: int = 60
) -> str:
    payload = {
        "model": PERPLEXITY_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Generate only the prompt text for this story.\n\n"
                    f"Story:\n{response_text}"
                ),
            },
        ],
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(
        PERPLEXITY_API_URL,
        headers=headers,
        data=json.dumps(payload),
        timeout=timeout_seconds,
    )
    resp.raise_for_status()
    print(resp.status_code)
    content = resp.json()["choices"][0]["message"]["content"].strip()
    # Keep only first line in case the model returns trailing text.
    return content.splitlines()[0].strip().strip('"')


def main() -> None:
    load_dotenv(dotenv_path=ENV_PATH)

    api_key = __import__("os").environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"Missing PERPLEXITY_API_KEY. Expected it in env or {ENV_PATH}."
        )

    tokenization_config = TokenizationConfig()
    instruction_tokens = InstructionTokensConfig()

    raw_data = IN_PATH.read_text(encoding="utf-8")
    readme_context = FINE_TUNING_README_PATH.read_text(encoding="utf-8")
    system_prompt = build_system_prompt(readme_context)

    responses = parse_examples(
        raw_data,
        tokenization_config.story_delimiter,
        instruction_tokens.prompt,
        instruction_tokens.response,
    )

    rebuilt_examples: list[str] = []
    for idx, response in enumerate(responses, start=1):
        prompt = generate_prompt_for_response(api_key, system_prompt, response)
        rebuilt_examples.append(
            f"{instruction_tokens.prompt}: {prompt}\n"
            f"{instruction_tokens.response}: {response}"
        )
        print(f"[{idx}/{len(responses)}] generated prompt")
        time.sleep(0.2)

    output_text = (
        f"\n{tokenization_config.story_delimiter}\n\n".join(rebuilt_examples)
        + f"\n{tokenization_config.story_delimiter}\n"
    )
    OUT_PATH.write_text(output_text, encoding="utf-8")
    print(f"Wrote {len(rebuilt_examples)} examples to {OUT_PATH}")


if __name__ == "__main__":
    main()
