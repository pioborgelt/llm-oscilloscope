# Subject Routing
This channel reads which broad subject subspace is active while the model generates and turns it into a routing measurement.

The useful question is whether a compact internal readout can stay informative while prompts, source datasets and model architectures change.

## The subject space

The current channel contains mathematics, physics, chemistry, biology, computer science, engineering, economics/business and psychology/social science.

The frozen core uses MMLU test and the non-MMLU-origin portion of MMLU-Pro test. Exact normalized question matches and MMLU-Pro rows marked as originating from MMLU were removed. The resulting core contains 1,664 unique questions, split evenly across sources and subjects.

Every question receives one of four subject-neutral templates. Separate unseen questions are reserved for template invariance and generation-time evaluation. The primary test trains on one source and evaluates on the other in both directions.

## Relation to existing work

I first found this subject separation independently during the broader discovery work for the Oscilloscope. However, I later found that the basic finding that scientific subjects occupy separable internal regions is not new. [*Large Language Models Encode Semantics and Alignment in Linearly Separable Representations*](https://arxiv.org/abs/2507.09709) studies 11 autoregressive models across six scientific topics and finds that high-level semantic information consistently forms compact, linearly separable subspaces. This is the closest direct external validation of the premise behind this channel.

A related Apple result is [*ExpertLens: Activation Steering Features Are Highly Interpretable*](https://machinelearning.apple.com/research/expertlens-activation). It makes a broader claim: concept-level activation features remain stable across models and datasets and recover human-like concept organization. Together, these results support the idea that useful semantic structure can be read from model activations instead of only inferred from the final text.

The part I am adding here is the instrument around that finding. This channel produces a graded subject reading after every generated token, tests it under source, template and fine-subject shift, and transports the frozen Llama readout to other model families without using their subject labels.


## During generation

I applied the head after every consumed token in 256 generations. Across 7k post-token states, it reaches 0.924 macro AUROC and averaging the probabilities over each prompt raises AUROC to 0.963.

One exploratory mixed-subject test asks the model to answer one question, emit a marker and answer a second question. The marker appeared in 26 of 32 generations, and the requested subject margin crossed at a median of zero tokens after it. Because the model controls the generated wording and marker coverage, this is a demonstration rather than part of the frozen pass rule.

## Transferring the channel
To move the channel, I run the same unlabeled prompts through Llama and the new model once and learn a small adapter between their internal spaces. No subject labels from the new model are needed.

On an untouched 512-row holdout, the frozen Llama channel reaches 0.933 macro AUROC on Mistral-7B and 0.934 on Qwen2.5-7B. With only 128 paired activations, it already reaches 0.875 and 0.895.

The compact verifier does not include this channel yet. The next test is transferring entity completion and support checking in the same way.
