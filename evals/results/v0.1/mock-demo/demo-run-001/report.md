# Story eval report: v0.1 / mock-demo

- run_id: demo-run-001
- total cases: 10
- cases passing all applicable deterministic evaluators: 9

## Deterministic summary by category

| category | total | overall_pass |
|---|---:|---:|
| basic-completion | 2 | 2 |
| prompt-adherence | 2 | 2 |
| tone | 2 | 2 |
| structural-constraints | 2 | 1 |
| robustness | 2 | 2 |

## Deterministic summary by evaluator

| evaluator | applicable cases | passed |
|---|---:|---:|
| clean_ending | 10 | 10 |
| entity_confusion | 10 | 9 |
| output_length | 10 | 10 |
| repetition | 10 | 10 |
| required_ending | 1 | 1 |
| required_lesson | 1 | 1 |
| required_opening | 2 | 2 |
| required_terms | 10 | 10 |
| weird_words | 10 | 10 |

## Cases


## basic-completion-mia-kite-001 (basic-completion)

**Overall deterministic result:** PASS

**Prompt:**

> Continue this story in a few sentences: Mia found a red kite behind the shed. She had never flown a kite before.

**Generated output:**

> Mia picked up the red kite and ran into the field. The wind caught it and lifted it high above the trees. "It flies!" she laughed, holding tight to the string until the sun began to set.

**Run metadata:**

| field | value |
|---|---|
| stop_reason | eos |
| latency_ms | 0.005 |
| prompt_tokens | 22 |
| completion_tokens | 37 |
| generation_seed | 3842942477 |
| model_version | mock-demo |
| run_id | demo-run-001 |
| tokenizer_id | mock-tokenizer |

**Code metrics:**

| evaluator | result |
|---|---|
| required_terms | PASS — `{'required': ['Mia', 'kite'], 'missing': []}` |
| required_opening | n/a — `{'expected': None}` |
| required_ending | n/a — `{'expected': None}` |
| required_lesson | n/a — `{}` |
| clean_ending | PASS — `{}` |
| repetition | PASS — `{'bigram_repetition_ratio': 0.0, 'trigram_repetition_ratio': 0.0}` |
| weird_words | PASS — `{'count': 0}` |
| entity_confusion | PASS — `{'unique_names': 1, 'names': ['Mia'], 'female_pronouns': 1, 'male_pronouns': 0, 'pronoun_mix': 0, 'adjacent_name_switches': 0, 'entity_confusion_score': 1}` |
| output_length | PASS — `{'word_count': 37, 'min_words': 15, 'max_words': 300}` |

**Human ratings (1-5):**

| dimension | rating | reviewer notes |
|---|---|---|
| Prompt adherence | _ | |
| Coherence and narrative flow | _ | |
| Child appropriateness | _ | |
| Overall story quality | _ | |

**Issue tag** (optional, one of repetition, lost-character, unsafe, ignored-constraint, unfinished): _

---

## basic-completion-sleepy-fox-002 (basic-completion)

**Overall deterministic result:** PASS

**Prompt:**

> Write a short story about a fox named Ru who cannot fall asleep.

**Generated output:**

> Ru the fox tossed and turned in his den. He counted stars through the window and listened to the crickets outside. Slowly his eyes grew heavy, and at last Ru fell asleep.

**Run metadata:**

| field | value |
|---|---|
| stop_reason | eos |
| latency_ms | 0.003 |
| prompt_tokens | 13 |
| completion_tokens | 32 |
| generation_seed | 1991293741 |
| model_version | mock-demo |
| run_id | demo-run-001 |
| tokenizer_id | mock-tokenizer |

**Code metrics:**

| evaluator | result |
|---|---|
| required_terms | PASS — `{'required': ['Ru'], 'missing': []}` |
| required_opening | n/a — `{'expected': None}` |
| required_ending | n/a — `{'expected': None}` |
| required_lesson | n/a — `{}` |
| clean_ending | PASS — `{}` |
| repetition | PASS — `{'bigram_repetition_ratio': 0.0, 'trigram_repetition_ratio': 0.0}` |
| weird_words | PASS — `{'count': 0}` |
| entity_confusion | PASS — `{'unique_names': 1, 'names': ['Slowly'], 'female_pronouns': 0, 'male_pronouns': 3, 'pronoun_mix': 0, 'adjacent_name_switches': 0, 'entity_confusion_score': 1}` |
| output_length | PASS — `{'word_count': 32, 'min_words': 15, 'max_words': 300}` |

**Human ratings (1-5):**

| dimension | rating | reviewer notes |
|---|---|---|
| Prompt adherence | _ | |
| Coherence and narrative flow | _ | |
| Child appropriateness | _ | |
| Overall story quality | _ | |

**Issue tag** (optional, one of repetition, lost-character, unsafe, ignored-constraint, unfinished): _

---

## prompt-adherence-lighthouse-003 (prompt-adherence)

**Overall deterministic result:** PASS

**Prompt:**

> Write a story set at a lighthouse by the sea. A boy named Theo visits his grandmother, who is the lighthouse keeper.

**Generated output:**

> Theo climbed the spiral stairs of the lighthouse to see his grandmother. She showed him how the great lamp turned to guide the ships home safely through the night.

**Run metadata:**

| field | value |
|---|---|
| stop_reason | eos |
| latency_ms | 0.002 |
| prompt_tokens | 22 |
| completion_tokens | 29 |
| generation_seed | 4119903117 |
| model_version | mock-demo |
| run_id | demo-run-001 |
| tokenizer_id | mock-tokenizer |

**Code metrics:**

| evaluator | result |
|---|---|
| required_terms | PASS — `{'required': ['Theo', 'lighthouse'], 'missing': []}` |
| required_opening | n/a — `{'expected': None}` |
| required_ending | n/a — `{'expected': None}` |
| required_lesson | n/a — `{}` |
| clean_ending | PASS — `{}` |
| repetition | PASS — `{'bigram_repetition_ratio': 0.0, 'trigram_repetition_ratio': 0.0}` |
| weird_words | PASS — `{'count': 0}` |
| entity_confusion | PASS — `{'unique_names': 1, 'names': ['Theo'], 'female_pronouns': 1, 'male_pronouns': 2, 'pronoun_mix': 1, 'adjacent_name_switches': 0, 'entity_confusion_score': 2}` |
| output_length | PASS — `{'word_count': 29, 'min_words': 15, 'max_words': 300}` |

**Human ratings (1-5):**

| dimension | rating | reviewer notes |
|---|---|---|
| Prompt adherence | _ | |
| Coherence and narrative flow | _ | |
| Child appropriateness | _ | |
| Overall story quality | _ | |

**Issue tag** (optional, one of repetition, lost-character, unsafe, ignored-constraint, unfinished): _

---

## prompt-adherence-blue-umbrella-004 (prompt-adherence)

**Overall deterministic result:** PASS

**Prompt:**

> Write a story about a girl named Priya who owns a blue umbrella that she believes is magic.

**Generated output:**

> Priya opened her blue umbrella even though the sky was clear. She twirled it once and pretended it could carry her over the rooftops, then folded it up and skipped home.

**Run metadata:**

| field | value |
|---|---|
| stop_reason | eos |
| latency_ms | 0.002 |
| prompt_tokens | 18 |
| completion_tokens | 31 |
| generation_seed | 3709852026 |
| model_version | mock-demo |
| run_id | demo-run-001 |
| tokenizer_id | mock-tokenizer |

**Code metrics:**

| evaluator | result |
|---|---|
| required_terms | PASS — `{'required': ['Priya', 'umbrella'], 'missing': []}` |
| required_opening | n/a — `{'expected': None}` |
| required_ending | n/a — `{'expected': None}` |
| required_lesson | n/a — `{}` |
| clean_ending | PASS — `{}` |
| repetition | PASS — `{'bigram_repetition_ratio': 0.0, 'trigram_repetition_ratio': 0.0}` |
| weird_words | PASS — `{'count': 0}` |
| entity_confusion | PASS — `{'unique_names': 1, 'names': ['Priya'], 'female_pronouns': 3, 'male_pronouns': 0, 'pronoun_mix': 0, 'adjacent_name_switches': 0, 'entity_confusion_score': 1}` |
| output_length | PASS — `{'word_count': 31, 'min_words': 15, 'max_words': 300}` |

**Human ratings (1-5):**

| dimension | rating | reviewer notes |
|---|---|---|
| Prompt adherence | _ | |
| Coherence and narrative flow | _ | |
| Child appropriateness | _ | |
| Overall story quality | _ | |

**Issue tag** (optional, one of repetition, lost-character, unsafe, ignored-constraint, unfinished): _

---

## tone-gentle-bedtime-005 (tone)

**Overall deterministic result:** PASS

**Prompt:**

> Write a gentle bedtime story about a rabbit named Nia who is getting ready for sleep. Start with 'Once upon a time'.

**Generated output:**

> Once upon a time, a little rabbit named Nia curled up in her soft burrow. She yawned, closed her eyes, and drifted off to a peaceful sleep under the quiet moonlight.

**Run metadata:**

| field | value |
|---|---|
| stop_reason | eos |
| latency_ms | 0.002 |
| prompt_tokens | 22 |
| completion_tokens | 31 |
| generation_seed | 2506482019 |
| model_version | mock-demo |
| run_id | demo-run-001 |
| tokenizer_id | mock-tokenizer |

**Code metrics:**

| evaluator | result |
|---|---|
| required_terms | PASS — `{'required': ['Nia'], 'missing': []}` |
| required_opening | PASS — `{'expected': 'Once upon a time'}` |
| required_ending | n/a — `{'expected': None}` |
| required_lesson | n/a — `{}` |
| clean_ending | PASS — `{}` |
| repetition | PASS — `{'bigram_repetition_ratio': 0.0, 'trigram_repetition_ratio': 0.0}` |
| weird_words | PASS — `{'count': 0}` |
| entity_confusion | PASS — `{'unique_names': 1, 'names': ['Nia'], 'female_pronouns': 3, 'male_pronouns': 0, 'pronoun_mix': 0, 'adjacent_name_switches': 0, 'entity_confusion_score': 1}` |
| output_length | PASS — `{'word_count': 31, 'min_words': 15, 'max_words': 300}` |

**Human ratings (1-5):**

| dimension | rating | reviewer notes |
|---|---|---|
| Prompt adherence | _ | |
| Coherence and narrative flow | _ | |
| Child appropriateness | _ | |
| Overall story quality | _ | |

**Issue tag** (optional, one of repetition, lost-character, unsafe, ignored-constraint, unfinished): _

---

## tone-silly-playful-006 (tone)

**Overall deterministic result:** PASS

**Prompt:**

> Write a silly, playful story about a puppy named Biscuit who tries to catch his own tail all afternoon.

**Generated output:**

> Biscuit the puppy spun in a circle, snapping at his own wagging tail. He tumbled over, popped back up, and chased it again, barking happily until he flopped down giggling in the grass.

**Run metadata:**

| field | value |
|---|---|
| stop_reason | eos |
| latency_ms | 0.002 |
| prompt_tokens | 19 |
| completion_tokens | 33 |
| generation_seed | 4046400653 |
| model_version | mock-demo |
| run_id | demo-run-001 |
| tokenizer_id | mock-tokenizer |

**Code metrics:**

| evaluator | result |
|---|---|
| required_terms | PASS — `{'required': ['Biscuit'], 'missing': []}` |
| required_opening | n/a — `{'expected': None}` |
| required_ending | n/a — `{'expected': None}` |
| required_lesson | n/a — `{}` |
| clean_ending | PASS — `{}` |
| repetition | PASS — `{'bigram_repetition_ratio': 0.0, 'trigram_repetition_ratio': 0.0}` |
| weird_words | PASS — `{'count': 0}` |
| entity_confusion | PASS — `{'unique_names': 1, 'names': ['Biscuit'], 'female_pronouns': 0, 'male_pronouns': 3, 'pronoun_mix': 0, 'adjacent_name_switches': 0, 'entity_confusion_score': 1}` |
| output_length | PASS — `{'word_count': 33, 'min_words': 15, 'max_words': 300}` |

**Human ratings (1-5):**

| dimension | rating | reviewer notes |
|---|---|---|
| Prompt adherence | _ | |
| Coherence and narrative flow | _ | |
| Child appropriateness | _ | |
| Overall story quality | _ | |

**Issue tag** (optional, one of repetition, lost-character, unsafe, ignored-constraint, unfinished): _

---

## structural-opening-ending-007 (structural-constraints)

**Overall deterministic result:** PASS

**Prompt:**

> Write a short story that starts with the exact words 'Once upon a time' and ends with the exact words 'The end.' about a turtle named Oto who is afraid of the water.

**Generated output:**

> Once upon a time, a turtle named Oto stood at the edge of the pond, too scared to dip in a toe. His friends cheered until he took one small step, then another, and finally swam with a splash. The end.

**Run metadata:**

| field | value |
|---|---|
| stop_reason | eos |
| latency_ms | 0.002 |
| prompt_tokens | 33 |
| completion_tokens | 41 |
| generation_seed | 104049366 |
| model_version | mock-demo |
| run_id | demo-run-001 |
| tokenizer_id | mock-tokenizer |

**Code metrics:**

| evaluator | result |
|---|---|
| required_terms | PASS — `{'required': ['Oto'], 'missing': []}` |
| required_opening | PASS — `{'expected': 'Once upon a time'}` |
| required_ending | PASS — `{'expected': 'The end.'}` |
| required_lesson | n/a — `{}` |
| clean_ending | PASS — `{}` |
| repetition | PASS — `{'bigram_repetition_ratio': 0.0, 'trigram_repetition_ratio': 0.0}` |
| weird_words | PASS — `{'count': 0}` |
| entity_confusion | PASS — `{'unique_names': 1, 'names': ['Oto'], 'female_pronouns': 0, 'male_pronouns': 2, 'pronoun_mix': 0, 'adjacent_name_switches': 0, 'entity_confusion_score': 1}` |
| output_length | PASS — `{'word_count': 41, 'min_words': 15, 'max_words': 300}` |

**Human ratings (1-5):**

| dimension | rating | reviewer notes |
|---|---|---|
| Prompt adherence | _ | |
| Coherence and narrative flow | _ | |
| Child appropriateness | _ | |
| Overall story quality | _ | |

**Issue tag** (optional, one of repetition, lost-character, unsafe, ignored-constraint, unfinished): _

---

## structural-lesson-008 (structural-constraints)

**Overall deterministic result:** FAIL

**Prompt:**

> Write a short story about a boy named Sam who does not want to share his toys, and end the story with the lesson he learns.

**Generated output:**

> Sam clutched his toys and turned away from his friends. Nobody wanted to play with him after that, and he felt lonely all afternoon. Sam learned that sharing his toys made playtime better for everyone.

**Run metadata:**

| field | value |
|---|---|
| stop_reason | eos |
| latency_ms | 0.002 |
| prompt_tokens | 26 |
| completion_tokens | 35 |
| generation_seed | 3623867704 |
| model_version | mock-demo |
| run_id | demo-run-001 |
| tokenizer_id | mock-tokenizer |

**Code metrics:**

| evaluator | result |
|---|---|
| required_terms | PASS — `{'required': ['Sam'], 'missing': []}` |
| required_opening | n/a — `{'expected': None}` |
| required_ending | n/a — `{'expected': None}` |
| required_lesson | PASS — `{'matched_phrases': ['learned']}` |
| clean_ending | PASS — `{}` |
| repetition | PASS — `{'bigram_repetition_ratio': 0.02941176470588236, 'trigram_repetition_ratio': 0.0}` |
| weird_words | PASS — `{'count': 0}` |
| entity_confusion | FAIL — `{'unique_names': 2, 'names': ['Nobody', 'Sam'], 'female_pronouns': 0, 'male_pronouns': 5, 'pronoun_mix': 0, 'adjacent_name_switches': 2, 'entity_confusion_score': 4}` |
| output_length | PASS — `{'word_count': 35, 'min_words': 15, 'max_words': 300}` |

**Human ratings (1-5):**

| dimension | rating | reviewer notes |
|---|---|---|
| Prompt adherence | _ | |
| Coherence and narrative flow | _ | |
| Child appropriateness | _ | |
| Overall story quality | _ | |

**Issue tag** (optional, one of repetition, lost-character, unsafe, ignored-constraint, unfinished): _

---

## robustness-multi-character-009 (robustness)

**Overall deterministic result:** PASS

**Prompt:**

> Write a short story about three friends, a bear named Otto, a mouse named Pip, and a duck named Wren, who build a raft together.

**Generated output:**

> Otto the bear, Pip the mouse, and Wren the duck gathered sticks by the river. Otto tied the logs together, Pip wove reeds to hold them tight, and Wren tested the raft by paddling it across the water.

**Run metadata:**

| field | value |
|---|---|
| stop_reason | eos |
| latency_ms | 0.002 |
| prompt_tokens | 25 |
| completion_tokens | 38 |
| generation_seed | 2894457580 |
| model_version | mock-demo |
| run_id | demo-run-001 |
| tokenizer_id | mock-tokenizer |

**Code metrics:**

| evaluator | result |
|---|---|
| required_terms | PASS — `{'required': ['Otto', 'Pip', 'Wren'], 'missing': []}` |
| required_opening | n/a — `{'expected': None}` |
| required_ending | n/a — `{'expected': None}` |
| required_lesson | n/a — `{}` |
| clean_ending | PASS — `{}` |
| repetition | PASS — `{'bigram_repetition_ratio': 0.027027027027026973, 'trigram_repetition_ratio': 0.0}` |
| weird_words | PASS — `{'count': 0}` |
| entity_confusion | PASS — `{'unique_names': 3, 'names': ['Otto', 'Pip', 'Wren'], 'female_pronouns': 0, 'male_pronouns': 0, 'pronoun_mix': 0, 'adjacent_name_switches': 0, 'entity_confusion_score': 3}` |
| output_length | PASS — `{'word_count': 38, 'min_words': 15, 'max_words': 300}` |

**Human ratings (1-5):**

| dimension | rating | reviewer notes |
|---|---|---|
| Prompt adherence | _ | |
| Coherence and narrative flow | _ | |
| Child appropriateness | _ | |
| Overall story quality | _ | |

**Issue tag** (optional, one of repetition, lost-character, unsafe, ignored-constraint, unfinished): _

---

## robustness-unusual-request-010 (robustness)

**Overall deterministic result:** PASS

**Prompt:**

> Write a short, child-safe story about a shy cloud named Puff who wants to learn how to whistle.

**Generated output:**

> Puff the cloud drifted quietly above the meadow, too shy to try whistling in front of the birds. One evening, alone above a quiet hill, Puff finally let out a soft, wobbly whistle and smiled.

**Run metadata:**

| field | value |
|---|---|
| stop_reason | eos |
| latency_ms | 0.001 |
| prompt_tokens | 18 |
| completion_tokens | 35 |
| generation_seed | 2436838951 |
| model_version | mock-demo |
| run_id | demo-run-001 |
| tokenizer_id | mock-tokenizer |

**Code metrics:**

| evaluator | result |
|---|---|
| required_terms | PASS — `{'required': ['Puff'], 'missing': []}` |
| required_opening | n/a — `{'expected': None}` |
| required_ending | n/a — `{'expected': None}` |
| required_lesson | n/a — `{}` |
| clean_ending | PASS — `{}` |
| repetition | PASS — `{'bigram_repetition_ratio': 0.0, 'trigram_repetition_ratio': 0.0}` |
| weird_words | PASS — `{'count': 0}` |
| entity_confusion | PASS — `{'unique_names': 1, 'names': ['Puff'], 'female_pronouns': 0, 'male_pronouns': 0, 'pronoun_mix': 0, 'adjacent_name_switches': 0, 'entity_confusion_score': 1}` |
| output_length | PASS — `{'word_count': 35, 'min_words': 15, 'max_words': 300}` |

**Human ratings (1-5):**

| dimension | rating | reviewer notes |
|---|---|---|
| Prompt adherence | _ | |
| Coherence and narrative flow | _ | |
| Child appropriateness | _ | |
| Overall story quality | _ | |

**Issue tag** (optional, one of repetition, lost-character, unsafe, ignored-constraint, unfinished): _

---
