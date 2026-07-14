"""Mocked generation backends implementing ``eval_suite.contracts.GenerateFn``.

Nothing here calls a real model. These stand in for whatever ``core/``
eventually exposes, satisfying the same minimal Protocol so the runner's
behavior can be tested without a live backend.
"""

from __future__ import annotations

from eval_suite.contracts import GenerationRequest, GenerationResult

# Canned, hand-written outputs keyed by case id. Each one is written to
# plausibly satisfy that case's expected_attributes, so a run against the
# real story-v0.1 dataset with this backend produces a mostly-passing report.
CANNED_OUTPUTS: dict[str, str] = {
    "basic-completion-mia-kite-001": (
        'Mia picked up the red kite and ran into the field. The wind caught '
        'it and lifted it high above the trees. "It flies!" she laughed, '
        "holding tight to the string until the sun began to set."
    ),
    "basic-completion-sleepy-fox-002": (
        "Ru the fox tossed and turned in his den. He counted stars through "
        "the window and listened to the crickets outside. Slowly his eyes "
        "grew heavy, and at last Ru fell asleep."
    ),
    "prompt-adherence-lighthouse-003": (
        "Theo climbed the spiral stairs of the lighthouse to see his "
        "grandmother. She showed him how the great lamp turned to guide the "
        "ships home safely through the night."
    ),
    "prompt-adherence-blue-umbrella-004": (
        "Priya opened her blue umbrella even though the sky was clear. She "
        "twirled it once and pretended it could carry her over the rooftops, "
        "then folded it up and skipped home."
    ),
    "tone-gentle-bedtime-005": (
        "Once upon a time, a little rabbit named Nia curled up in her soft "
        "burrow. She yawned, closed her eyes, and drifted off to a peaceful "
        "sleep under the quiet moonlight."
    ),
    "tone-silly-playful-006": (
        "Biscuit the puppy spun in a circle, snapping at his own wagging "
        "tail. He tumbled over, popped back up, and chased it again, "
        "barking happily until he flopped down giggling in the grass."
    ),
    "structural-opening-ending-007": (
        "Once upon a time, a turtle named Oto stood at the edge of the pond, "
        "too scared to dip in a toe. His friends cheered until he took one "
        "small step, then another, and finally swam with a splash. The end."
    ),
    "structural-lesson-008": (
        "Sam clutched his toys and turned away from his friends. Nobody "
        "wanted to play with him after that, and he felt lonely all "
        "afternoon. Sam learned that sharing his toys made playtime better "
        "for everyone."
    ),
    "robustness-multi-character-009": (
        "Otto the bear, Pip the mouse, and Wren the duck gathered sticks by "
        "the river. Otto tied the logs together, Pip wove reeds to hold them "
        "tight, and Wren tested the raft by paddling it across the water."
    ),
    "robustness-unusual-request-010": (
        "Puff the cloud drifted quietly above the meadow, too shy to try "
        "whistling in front of the birds. One evening, alone above a quiet "
        "hill, Puff finally let out a soft, wobbly whistle and smiled."
    ),
}


class CannedGenerator:
    """Returns a fixed, hand-written completion per case id.

    Looked up via the prompt text since ``GenerationRequest`` does not carry
    a case id (see ``evals/CORE_INTEGRATION_CONTRACT.md`` for why); the runner
    is expected to be exercised one dataset at a time, so prompt text is a
    stable enough key for this fixture.
    """

    def __init__(self, outputs_by_prompt: dict[str, str]):
        self._outputs_by_prompt = outputs_by_prompt

    def __call__(self, request: GenerationRequest) -> GenerationResult:
        text = self._outputs_by_prompt[request.prompt]
        return GenerationResult(
            text=text,
            stop_reason="eos",
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=len(text.split()),
        )


def canned_generator_for(cases: list) -> CannedGenerator:
    outputs_by_prompt = {case.prompt: CANNED_OUTPUTS[case.id] for case in cases}
    return CannedGenerator(outputs_by_prompt)


class DeterministicSeedEchoGenerator:
    """Minimal generator whose output is a pure function of the request.

    Used to test runner determinism/reproducibility independent of any
    hand-written story text: the same request must always produce the same
    ``GenerationResult``.
    """

    def __call__(self, request: GenerationRequest) -> GenerationResult:
        text = f"[seed={request.seed}] {request.prompt}"
        return GenerationResult(
            text=text,
            stop_reason="eos",
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=len(text.split()),
        )


class BrokenGenerator:
    """Generator that violates constraints, for testing failure detection."""

    def __call__(self, request: GenerationRequest) -> GenerationResult:
        return GenerationResult(
            text="the the the the the the",
            stop_reason="max_tokens",
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=6,
        )
