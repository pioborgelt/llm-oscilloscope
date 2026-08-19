# Subject Routing

The Subject Routing channel estimates which broad academic subject is active
inside the model while it generates. Instead of forcing one hard category, it
returns a graded distribution over mathematics, physics, chemistry, biology,
computer science, engineering, economics/business and psychology/social
science.

This channel is independent from the unsupported-entity detector. Its purpose is to make one part of the model's
internal routing visible after every token.

## Why this channel?

The Oscilloscope is meant to contain several narrow readings instead of one
general confidence score. Subject Routing shows which knowledge region appears
to be active and could later help select specialized tools, retrieval sources
or interventions.

The basic observation that subjects are linearly separable in model activations
is not new. [*Large Language Models Encode Semantics and Alignment in Linearly
Separable Representations*](https://arxiv.org/abs/2507.09709) finds similar
high-level semantic regions across several model families. What I am adding is
a tokenwise measurement channel and a practical way to move it between models.

## What I tested

I trained the original channel on Llama using questions from two different
academic benchmarks. Training and testing were separated by source, with
additional unseen templates and fine subjects used as controls. The main
cross-source result is **0.878 macro AUROC**.


## Moving it to other models

To move the channel, I run the same unlabeled prompts through Llama and the new
model once, then fit a small adapter between their internal spaces. No subject
labels from the new model are used.

On untouched questions, the unchanged channel reaches **0.933 macro AUROC on
Mistral-7B** and **0.934 on Qwen2.5-7B**. It also remains informative on Qwen's
own generated-token stream. More importantly, every held-out subject pair stays
above chance even when those subjects were removed during adapter onboarding.

This makes the channel relatively cheap to onboard, but it does not mean that
arbitrary internal signals can be transferred between arbitrary models.

## Limits

The current subject space contains only eight broad academic areas and has been
tested mainly on short English multiple-choice questions. It is not yet a
general router for multilingual text, code or conversations that mix several
domains.

Adapters fitted only on broad factual prompts missed the frozen transfer
threshold. This means the onboarding data still needs to cover the relevant
domain and point in generation. The output should also be treated as a graded
ranking, not a deterministic subject label.

The portable heads, frozen predictions and failed controls are included under
[`artifacts/subject_routing/`](../../artifacts/subject_routing/). They can be
checked with:

```
python scripts/verify_subject_routing.py
```
