# Retrieval Timing

This channel measures a state change around the transition from reading a factual question to beginning the answer. It is one of the oldest channels in this project and also the one whose interpretation changed the most.

I initially treated it as a truth or verification signal. Stronger controls showed that this interpretation was too broad, because the same movement can appear when factual retrieval fires and returns the wrong fact. The better interpretation is retrieval timing: did the model enter the characteristic internal transition associated with factual lookup?

## The measurement

The basic feature is:

```
V_dyn = h_gen1 - h_trigger
```

`h_trigger` is the residual state at the final prompt token, and `h_gen1` is the state after the first generated token. On Llama-3.1-8B-Instruct, the useful transition is concentrated around layers 12 to 15.

I first tested this channel on 10,017 prompts across eight different factual-QA datasets listed in the appendix. Across that set, it consistently captures an early change between processing the question and beginning the answer. For the Oscilloscope, this tells us whether the model enters its usual retrieval-like transition before an answer entity starts to unfold. This gives us an earlier measurement than entity completion and support checking.

## What the direction contains

Most discriminative power collapses into one direction. It explains 7.1% of the variance in `V_dyn` and reaches 0.903 AUROC on the original correct-versus-error separation.

However, correct and wrong retrieval can occupy the same side of this direction, and the strong separation mainly comes from cases where the retrieval-like transition does not fire and the model instead echoes or copies part of the question.

## Confound controls

Unit-normalizing V_dyn leaves the AUROC at 0.915, while vector norm alone reaches only 0.554. Removing response length, prompt-answer cosine, hidden-state norms and source indicators from the score still leaves 0.833 AUROC.

But unfortunately, the channel is not equally useful everywhere. Date-like "when" questions are strongest, while person-name "who" questions form a clear blind spot. In the person-name slice the direction is at chance, with 0.504 AUROC. This is also why `V_dyn` cannot replace the later support channel.

## Cross-model transport

The retrieval-timing feature was also used in the earlier four-model transfer experiment with Llama-3.1-8B, Phi-3.5-mini, Mistral-7B and Qwen2.5-7B.

Across all 12 directed model pairs, `V_dyn1` transport reaches 0.752 mean AUROC under question-hash cross-validation. Every pair is above chance, the weakest is 0.695 and shuffled pairing averages 0.499. Leave-one-dataset-out reaches 0.677 mean AUROC, with a weakest pair of 0.651 and shuffled mean 0.502.

The result therefore supports channel transport on unseen questions, but it does not show that dynamic features are universally better, and it does not turn answer-string labels into retrieval ground truth.

## How this channel should be used

Retrieval timing is near the start of the generation lifecycle. Entity completion and support checking operate later, so this additional channel can distinguish a case where retrieval never started from one where the retrieved information looks unsupported.

It should not be treated as a direct steering vector. Earlier interventions at the retrieval layer produced no reliable factual flips.

## What is and is not released here

The full retrieval caches, mechanistic scripts and strict transfer arrays remain in the research workspace. They are not part of the compact detector verifier repository here.

The next decisive experiment is to train a clean retrieval-event target rather than reusing answer correctness, then transport that frozen channel with aligned trigger and generation endpoints. Until then, this is a strong mechanistic lead and a working research channel.


## Appendix
Datasets used for the 10k Run: TriviaQA, SciQ, WebQuestions, NaturalQuestions, MedQA, MMLU, TruthfulQA and TriviaQA-MC
