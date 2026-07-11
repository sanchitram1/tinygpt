import random
from dataclasses import dataclass

from config import DataConfig, TokenizationConfig
from tokenizer import iter_stories

random.seed(42)


@dataclass
class InstructionTokensConfig:
    prompt: str = "<prompt>"
    response: str = "<response>"


tokenization_config = TokenizationConfig()
data_config = DataConfig()
instruction_tokens = InstructionTokensConfig()
SAMPLE = 100


def main():
    all_stories: list[str] = []
    starters = [
        "Write a short children's story about ...",
        "Tell me a children's story about ...",
        "Give me a story featuring ...",
        "Write a bedtime story about ...",
    ]

    for i, training_story in enumerate(
        iter_stories(tokenization_config, data_config.validation_file_local)
    ):
        all_stories.append(
            f"{instruction_tokens.prompt}: {starters[i % 4]}\n{instruction_tokens.response}: {training_story}"
        )

    print(f"Total stories to choose from: {len(all_stories)}")
    print(f"{all_stories[-1]}")

    instructions = random.sample(all_stories, SAMPLE)

    with open(f"{data_config.instruction_validation_file}", "w") as f:
        f.write(f"\n{tokenization_config.story_delimiter}\n\n".join(instructions))

    print(f"Saved to {data_config.instruction_validation_file}")


if __name__ == "__main__":
    main()
