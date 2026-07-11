"""generation_quality.py must keep its CLI behavior after extracting its
metric functions into eval_suite.quality_metrics (plan step 2)."""

from eval_suite import quality_metrics
import generation_quality


SAMPLE_TEXT = 'Nia the rabbit said, "Good night!" and she went to sleep.'
REPETITIVE_TEXT = "the the the the the the"


def test_generation_quality_private_aliases_match_shared_module():
    assert generation_quality._words(SAMPLE_TEXT) == quality_metrics.words(SAMPLE_TEXT)
    assert generation_quality._bigram_repetition_ratio(
        REPETITIVE_TEXT
    ) == quality_metrics.bigram_repetition_ratio(REPETITIVE_TEXT)
    assert generation_quality._trigram_repetition_ratio(
        REPETITIVE_TEXT
    ) == quality_metrics.trigram_repetition_ratio(REPETITIVE_TEXT)
    assert generation_quality._ends_cleanly(SAMPLE_TEXT) == quality_metrics.ends_cleanly(
        SAMPLE_TEXT
    )
    assert generation_quality._entity_metrics(SAMPLE_TEXT) == quality_metrics.entity_metrics(
        SAMPLE_TEXT
    )


def test_summarize_generations_cli_behavior_unchanged():
    generations = [
        {
            "model": "xlarge-plus",
            "prompt": "Once upon a time",
            "temperature": 0.7,
            "top_k": 30,
            "generated_text": SAMPLE_TEXT,
        },
        {
            "model": "xlarge-plus",
            "prompt": "A fox",
            "temperature": 0.7,
            "top_k": 30,
            "generated_text": REPETITIVE_TEXT,
        },
    ]
    summary = generation_quality.summarize_generations(generations)
    assert summary["total_generations"] == 2
    assert summary["models"] == {"xlarge-plus": 2}
    assert summary["by_model"]["xlarge-plus"]["n"] == 2
    assert summary["by_model"]["xlarge-plus"]["clean_endings"] == 1
