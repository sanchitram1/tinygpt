# Fine-Tuning TinyStories

This folder contains the first steps toward turning our pretrained TinyStories
language model into something a little more instruction-following and
chatbot-like.

The base `hw3` model was trained with standard next-token prediction on raw
story text. That gives us a decent small generative model, but it does not
really understand prompts in the way an instruction-tuned model does. The goal
here is to build a small supervised fine-tuning dataset and then use LoRA to
adapt the pretrained model without retraining the full network from scratch.

## What We Are Doing

We are creating a new instruction-style dataset where each example looks like:

```txt
<prompt>: Tell me a children's story about a shy rabbit learning patience.
<response>: Once upon a time, there was a shy rabbit...
<|endoftext|>
```

This keeps the task in the same domain as pretraining, short children's
stories, while changing the format from plain story continuation to
prompt-and-response behavior.

The high-level idea is:

1. Start from the pretrained TinyStories model.
2. Sample real stories from the original training corpus.
3. Wrap those stories in an instruction-response format.
4. Manually edit the prompts so they actually describe the story content.
5. Fine-tune the model on this new dataset using LoRA.

## Why This Helps

During pretraining, the model sees raw text and learns general story structure,
grammar, and local continuation patterns. It does not explicitly learn that a
user prompt should be answered with a helpful response.

By building prompt-response examples, we teach the model a new behavior:

- read a request
- recognize the task as "write a story"
- produce the story as the response

This is a small version of supervised fine-tuning (SFT). The hope is not to
turn the model into a general-purpose assistant, but to make it more responsive
to explicit instructions in the TinyStories domain.

## Current Dataset Approach

The script [`build_instruction_dataset.py`](./build_instruction_dataset.py)
samples stories from the original TinyStories training text and writes out a new
instruction-format dataset.

Right now the workflow is:

1. Read stories from the original corpus with `iter_stories(...)`.
2. Sample a fixed number of stories from the training set.
3. Prepend each story with a templated `<prompt>` field.
4. Store the original story under `<response>`.
5. Separate examples with the same story delimiter used elsewhere in the
   homework pipeline.

The prompt templates are intentionally generic at first, for example:

- `Write a short children's story about ...`
- `Tell me a children's story about ...`
- `Give me a story featuring ...`
- `Write a bedtime story about ...`

After sampling, the important manual step is to replace the `...` with a short,
content-aware description of the actual story. That is what makes the dataset a
true instruction-following dataset instead of just a formatting exercise.

## Why Sample Existing Stories Instead of Generating New Ones

We are using stories from the original TinyStories text instead of model
generations because the original data is cleaner and more stable. If we trained
on the model's own raw generations, we would risk reinforcing its current
mistakes.

Using sampled ground-truth stories gives us:

- higher-quality targets
- less noise
- a simpler pipeline
- a cleaner comparison between pretraining and fine-tuning

## Planned Fine-Tuning Method: LoRA

We want to learn LoRA, so the plan is to fine-tune the pretrained checkpoint
with low-rank adapters rather than updating every model weight.

The expected flow is:

1. Load the pretrained TinyGPT checkpoint.
2. Freeze the original model weights.
3. Add LoRA adapters to selected linear layers, especially attention layers.
4. Train only the adapter parameters on the instruction dataset.

This keeps the experiment lightweight and makes it easier to study how a small
behavioral change can be added on top of a pretrained model.

## Scope and Expectations

This is a small experiment with a small model and a small dataset. The goal is
not to build a fully capable chat model. The goal is to see whether a pretrained
story model can become more instruction-following in a narrow setting:

- user gives a story-writing prompt
- model responds in a clearer prompt/response format
- model behaves a bit more like a tiny domain-specific chatbot

If this works, the next step would be to compare generations before and after
fine-tuning and see whether the model follows explicit prompts more reliably.
