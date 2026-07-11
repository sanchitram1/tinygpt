from eval_suite.case_evaluators import evaluate_case
from eval_suite.schema import DatasetCase


def make_case(**expected_attributes) -> DatasetCase:
    return DatasetCase(
        id="test-case-001",
        prompt="Write a story.",
        category="basic-completion",
        expected_attributes={"child_safe": True, **expected_attributes},
    )


def test_required_terms_pass_when_all_present():
    case = make_case(required_terms=["Nia", "rabbit"])
    result = evaluate_case(case, "Nia the rabbit went to sleep.")
    assert result["required_terms"]["pass"] is True
    assert result["required_terms"]["missing"] == []


def test_required_terms_fail_when_missing():
    case = make_case(required_terms=["Nia", "rabbit"])
    result = evaluate_case(case, "The fox went to sleep.")
    assert result["required_terms"]["pass"] is False
    assert result["required_terms"]["missing"] == ["Nia", "rabbit"]


def test_required_terms_not_applicable_when_unset():
    case = make_case()
    result = evaluate_case(case, "Anything goes here.")
    assert result["required_terms"]["applicable"] is False
    assert result["required_terms"]["pass"] is None


def test_required_opening_pass():
    case = make_case(required_opening="Once upon a time")
    result = evaluate_case(case, "Once upon a time there was a rabbit.")
    assert result["required_opening"]["pass"] is True


def test_required_opening_fail():
    case = make_case(required_opening="Once upon a time")
    result = evaluate_case(case, "There was a rabbit.")
    assert result["required_opening"]["pass"] is False


def test_required_ending_pass():
    case = make_case(required_ending="The end.")
    result = evaluate_case(case, "The rabbit went to sleep. The end.")
    assert result["required_ending"]["pass"] is True


def test_required_ending_fail():
    case = make_case(required_ending="The end.")
    result = evaluate_case(case, "The rabbit went to sleep.")
    assert result["required_ending"]["pass"] is False


def test_required_lesson_pass_on_matched_phrase():
    case = make_case(required_lesson=True)
    result = evaluate_case(case, "Sam learned that sharing was better than keeping everything.")
    assert result["required_lesson"]["pass"] is True
    assert "learned" in result["required_lesson"]["matched_phrases"]


def test_required_lesson_fail_without_phrase():
    case = make_case(required_lesson=True)
    result = evaluate_case(case, "Sam played with his toys all day.")
    assert result["required_lesson"]["pass"] is False


def test_clean_ending_pass_and_fail():
    case = make_case()
    assert evaluate_case(case, "It was a good day.")["clean_ending"]["pass"] is True
    assert evaluate_case(case, "It was a good day")["clean_ending"]["pass"] is False


def test_repetition_flags_repeated_text():
    case = make_case()
    result = evaluate_case(case, "the the the the the the the the the the")
    assert result["repetition"]["pass"] is False
    assert result["repetition"]["bigram_repetition_ratio"] > 0


def test_repetition_passes_for_varied_text():
    case = make_case()
    text = (
        "Mia found a red kite behind the shed. She had never flown one "
        "before, but the wind lifted it gently into the bright blue sky."
    )
    result = evaluate_case(case, text)
    assert result["repetition"]["pass"] is True


def test_entity_confusion_multiple_characters_pass():
    case = make_case(multiple_characters=True, required_terms=["Otto", "Pip", "Wren"])
    text = "Otto the bear, Pip the mouse, and Wren the duck built a raft together."
    result = evaluate_case(case, text)
    assert result["entity_confusion"]["pass"] is True
    assert result["entity_confusion"]["unique_names"] == 3


def test_entity_confusion_multiple_characters_fail_when_too_few_names():
    case = make_case(multiple_characters=True, required_terms=["Otto", "Pip", "Wren"])
    text = "Otto built a raft alone."
    result = evaluate_case(case, text)
    assert result["entity_confusion"]["pass"] is False


def test_output_length_too_short_fails():
    case = make_case()
    result = evaluate_case(case, "Too short.")
    assert result["output_length"]["pass"] is False


def test_output_length_within_range_passes():
    case = make_case()
    text = " ".join(["word"] * 50) + "."
    result = evaluate_case(case, text)
    assert result["output_length"]["pass"] is True


def test_overall_pass_is_true_when_all_applicable_evaluators_pass():
    case = make_case(required_terms=["Nia"], required_opening="Once upon a time")
    text = (
        "Once upon a time, a rabbit named Nia curled up in her burrow and "
        "drifted off to a peaceful sleep under the quiet evening sky."
    )
    result = evaluate_case(case, text)
    assert result["overall_pass"] is True


def test_overall_pass_is_false_when_any_applicable_evaluator_fails():
    case = make_case(required_terms=["Nia"])
    result = evaluate_case(case, "There was a fox.")
    assert result["overall_pass"] is False


def test_overall_pass_is_none_when_no_applicable_evaluators_have_a_verdict():
    # clean_ending, repetition, weird_words, entity_confusion, and
    # output_length are always applicable, so overall_pass is never None in
    # practice -- this documents that guarantee explicitly.
    case = make_case()
    result = evaluate_case(case, "A reasonably long and clean sentence about a fox.")
    assert result["overall_pass"] is not None
