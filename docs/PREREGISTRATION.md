# Completion-Aware Online Entity Detector: Preregistration

Date frozen: 2026-07-12, before the first completion-aware end-to-end result.

## Question

Does entity-completion information improve the existing online answer-entity hallucination cascade when no oracle entity positions are available at test time?

## Fixed Instrument

- Boundary channel: a linear L24-post head predicts whether an answer entity ends at the current generated token.
- Correctness channel: a tokenwise linear L30-post head predicts whether the emitted answer entity is unsupported.
- Primary score at every generated token: `P(answer_entity_end) * P(unsupported)`.
- Test-time inputs: hidden states and generated-token probabilities only. Gold spans are used only to construct held-out evaluation labels.

The boundary layer is fixed from the existing entity-channel result. The correctness layer and post-token timing are fixed from the completed first/last/span-max control. There is no test-set layer or aggregator search.

## Data And Labels

- P15 Llama-3.1-8B-Instruct cache.
- Dual-judge consensus correctness labels.
- Positive event: the final token of an aligned `ANSWER + UNSUPPORTED` entity.
- Negative event: every other scorable generated token, including supported answer-entity ends.
- Ignored: answer-entity ends with partial, disputed, unverifiable, unaligned, or otherwise unknown correctness.
- Rows without a cached post-token state are excluded because their absence is an extraction-limit artifact.

## Evaluation

Outer splits are fixed to:

1. Prompt-grouped random 5-fold.
2. Answer-entity-disjoint 5-fold, with zero normalized answer-entity overlap.
3. Leave-one-source-out.

Each outer training partition is split again into fit and validation prompts. Model fitting and correctness-head regularization selection use fit prompts only; regularization is selected on an additional fit-internal split using final-token entity AUROC. Calibration and operating thresholds use validation prompts only.

Primary metrics:

- Natural-token-stream average precision and AUROC.
- Precision and recall at a validation-selected target near one false alarm per 100 tokens.
- Exact entity-end detection AP/F1.
- Alert delay relative to the gold entity end.

Fixed controls:

- Boundary channel only.
- Correctness channel only.
- Official arXiv:2509.03531 L30 head, alone and combined with the learned boundary channel.
- Token surprisal.
- Existing cleaned-label onset cascade as the historical baseline.

## Hypothesis And Stopping Rule

The completion-aware product should exceed the cleaned onset cascade's random AP of 0.279 and remain stable under answer-entity-disjoint and leave-one-source-out evaluation. Regardless of outcome, the three fixed split evaluations and controls close this experiment; no post-hoc layer sweep will be added.
