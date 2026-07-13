# Matched Forced-Answer External Holdout v2: Preregistration

Date frozen: 2026-07-13, before generating any v2 model outputs.

## Motivation and status

This is a new follow-up designed after diagnosing the v1 stress test. It does not
replace v1. The v1 PopQA/HotpotQA run had only 29.5% answered coverage and used
HotpotQA without its intended context. V2 asks a narrower question: does the frozen
correctness head transfer when two new short-form factual-QA sources actually
produce candidate answers?

Because the task design was informed by v1, v2 must be reported as a matched
follow-up rather than as the originally planned external confirmation.

## Frozen data

- 800 questions from OpenAI SimpleQA's official test CSV.
- 800 questions from Google's GRANOLA Entity Questions CSV.
- GRANOLA eligibility is fixed before generation: `score_for_potential_error == 0`,
  a non-empty answer of at most eight words, and question-entity popularity at or
  above the median among otherwise eligible rows.
- Selection within each source uses SHA256 ranking with seed
  `p15-matched-holdout-v2-2026-07-13`.
- Exact normalized questions in P12/P15 or external holdout v1 are excluded before
  ranking. Gold-answer surface overlap is reported descriptively and is not used
  to select or remove v2 examples.
- GRANOLA's original answer and all non-empty multi-granularity answers are supplied
  as authoritative aliases.

## Frozen generation

- Model: `meta-llama/Llama-3.1-8B-Instruct`, 8-bit.
- Greedy decoding, maximum 16 generated tokens.
- Standard chat template.
- Exact user instruction:

  `Answer the question with your single best short factual answer. Give one answer even if uncertain. Do not explain, hedge, or mention uncertainty.`

  followed by the dataset question.

No prompt variant may be selected after seeing generations.

## Frozen instrument

- Existing `release_head.npz` from commit `52bdb7c`.
- L24 post-token entity-boundary head and L30 post-token correctness head.
- No fitting, layer selection, recalibration, threshold selection, or weight update.
- Primary score: frozen L30 unsupported probability at independently labeled
  answer-entity final tokens.
- Full-stream boundary-correctness product is secondary.

## Blind labels

DeepSeek V4 Flash and Qwen 3.5 Flash independently receive question, gold aliases,
generation, and token IDs, but no detector scores. A primary label survives only
when both agree on `ANSWER`, final token, and `SUPPORTED` or `UNSUPPORTED`.
Disagreements, malformed responses, partial answers, unverifiable answers, and
unaligned spans are excluded and reported. IDK responses remain abstentions and
are never counted as correct, even though the prompt requests an answer.

## Confirmatory endpoints

Primary:

1. Pooled L30 entity-end AUROC and AP with 2,000 prompt-bootstrap resamples.
2. SimpleQA and GRANOLA entity-end AUROC/AP separately.

Secondary:

1. Answered coverage, abstention, ambiguity, and judge agreement.
2. Natural-token-stream product AUROC/AP for all valid and answered-only prompts.
3. Boundary-only entity-end localization.
4. Fixed product threshold `>= 0.5` precision/recall, without v2 tuning.
5. Exact question/answer overlap and a clearly post-hoc answer-overlap-excluded
   sensitivity analysis.

## Decision and stopping rule

The matched transfer criterion passes only if the pooled entity-end AUROC bootstrap
lower bound exceeds 0.5 and neither source point AUROC is below 0.5. All outcomes,
coverage, exclusions, per-source metrics, and confidence intervals will be kept
internally. No third dataset, prompt variant, layer, calibration, or threshold will
be tried based on v2 results.
