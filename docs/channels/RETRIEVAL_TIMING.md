# Retrieval Timing

Status: active research and release candidate; not a released measurement
channel. Several negative readout and holdout results delayed its release.

This work studies a state change around factual retrieval and the transition
into an answer. It is one of the oldest channel candidates in this project and
also the one whose interpretation changed the most. Also, it is the channel I probably understand the least.

I initially treated it as a truth or verification signal. Stronger controls showed that this interpretation was too broad, because the same movement can appear when factual retrieval fires and returns the wrong fact. The better interpretation is retrieval timing: did the model enter the characteristic internal transition associated with factual lookup?

## Measurement

The basic feature is:

```
V_dyn = h_gen1 - h_trigger
```

h_trigger is the residual state at the final prompt token, and h_gen1 is the state after the first generated token. On Llama-3.1-8B-Instruct, the useful transition is concentrated around layers 12 to 15.

I first tested this feature on 10,017 prompts across eight different factual-QA
datasets listed in the appendix. It ranks an early change between processing
the question and beginning the answer, but those answer-derived labels do not
establish a clean online retrieval event. V_dyn is therefore retained as a
historical retrieval-timing baseline rather than the current channel.

## What the direction contains

Most discriminative power collapses into one direction. It explains 7.1% of the variance in V_dyn and reaches 0.903 AUROC on the original correct-versus-error separation.

However, correct and wrong retrieval can occupy the same side of this direction, and the strong separation mainly comes from cases where the retrieval-like transition does not fire and the model instead echoes or copies part of the question.

## Confound controls

Unit-normalizing V_dyn leaves the AUROC at 0.915, while vector norm alone reaches only 0.554. Removing response length, prompt-answer cosine, hidden-state norms and source indicators from the score still leaves 0.833 AUROC.

But unfortunately, the channel is not equally useful everywhere. Date-like "when" questions are strongest, while person name "who" questions form a clear blind spot. In the person name slice the direction is at chance, with 0.504 AUROC. This is also why V_dyn cannot replace the later support channel.

## Cross-model transport

The retrieval-timing feature was also used in the earlier four-model transfer experiment with Llama-3.1-8B, Phi-3.5-mini, Mistral-7B and Qwen2.5-7B.

Across all 12 directed model pairs, V_dyn1 transport reaches 0.752 mean AUROC under question-hash cross-validation. Every pair is above chance, the weakest is 0.695 and shuffled pairing averages 0.499. Leave-one-dataset-out reaches 0.677 mean AUROC, with a weakest pair of 0.651 and shuffled mean 0.502.

The result supports transport of this historical feature on unseen questions, but it does not show that dynamic features are universally better, does not turn answer-string labels into retrieval ground truth and does not by itself establish a releasable retrieval channel.


## Intended role

Retrieval timing belongs near the start of the generation lifecycle, while
entity completion and support checking operate later. A validated version could
eventually help distinguish failed retrieval from a later unsupported answer,
but the current public evidence does not yet establish that online distinction.

It should not be treated as a direct steering vector. Earlier interventions at the retrieval layer produced no reliable factual flips.


## Appendix
Datasets used for the 10k Run: TriviaQA, SciQ, WebQuestions, NaturalQuestions, MedQA, MMLU, TruthfulQA and TriviaQA-MC
