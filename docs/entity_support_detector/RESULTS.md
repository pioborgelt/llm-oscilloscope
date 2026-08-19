# Completion-Aware Per-Token Entity Detection

The evaluated score is `P(answer_entity_end at L24-post) * P(unsupported at L30-post)`. The detector produces it after every generated token and does not receive gold entity positions at test time.

Of 164,203 labeled-stream tokens, 159,824 have both required post-token states in the cache. The missing rows come from the extraction limit and were removed before evaluation.

## Main Result

| Split | Old onset AUROC/AP | Completion AUROC/AP | AP delta | AP/prevalence |
|---|---:|---:|---:|---:|
| random | 0.965 / 0.279 | 0.983 / 0.420 | +0.140 | 41.1x |
| entity_disjoint | 0.965 / 0.272 | 0.982 / 0.417 | +0.145 | 40.8x |
| loso | 0.959 / 0.252 | 0.980 / 0.391 | +0.140 | 38.3x |

## Controls On The Random Split

| Method | AUROC/AP |
|---|---:|
| Learned boundary x learned correctness | 0.983 / 0.420 |
| Learned boundary x official correctness | 0.978 / 0.347 |
| Boundary only | 0.975 / 0.229 |
| Official correctness only | 0.811 / 0.095 |
| Learned correctness only | 0.623 / 0.022 |
| Token surprisal | 0.430 / 0.009 |

## Paired Prompt Bootstrap

- Completion - old onset AP: +0.141 (95% CI +0.114 to +0.167; p < 0.001).
- Completion - boundary x official AP: +0.073 (95% CI +0.054 to +0.092; p < 0.001).

## One-False-Alarm Operating Point

| Split | Old P/R | Completion P/R | False alarms / 100 tokens |
|---|---:|---:|---:|
| random | 0.304/0.412 | 0.368/0.557 | 0.977 |
| entity_disjoint | 0.300/0.402 | 0.367/0.569 | 1.002 |
| loso | 0.286/0.377 | 0.352/0.542 | 1.020 |

## Entity Ending And Alert Delay

On the random folds, the answer-entity-end head reaches 0.989 AUROC, 0.859 AP and a validation-selected F1 of 0.778.

At the operating point near one false alarm per 100 tokens, windowed unsupported-entity recall is 0.585. Median alert delay is zero tokens in every random fold, so the median alert lands on the entity's final token rather than later.

## Leave-One-Source-Out

| Held-out source | Positive events | AUROC | AP | AP/prevalence |
|---|---:|---:|---:|---:|
| medqa | 33 | 0.993 | 0.330 | 118.9x |
| nq | 559 | 0.975 | 0.399 | 29.6x |
| sciq | 298 | 0.986 | 0.466 | 39.0x |
| triviaqa | 341 | 0.982 | 0.422 | 42.6x |
| triviaqa_mc | 59 | 0.987 | 0.208 | 52.6x |
| webq | 321 | 0.977 | 0.391 | 36.6x |

## What The Result Says

Waiting for entity completion improves both ranking and the fixed false-alarm operating point compared with the old onset detector. The gain remains under answer-entity-disjoint and leave-one-source-out evaluation. The learned L30 correctness channel also adds information beyond entity completion alone and beyond the published L30 head.

This is still a result on short factual QA. The detector sees an entity error only when the entity has already ended, and the operating point is not strong enough for deployment.

## More Detail

- [Entity-completion results](ENTITY_COMPLETION_RESULTS.md) compare the first and last entity token and include the paired prompt bootstrap.
- The machine-readable matched-holdout artifact covers the frozen-head SimpleQA/GRANOLA transfer check and its source-imbalance problem.
