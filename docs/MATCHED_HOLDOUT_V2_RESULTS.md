# Matched Forced-Answer External Holdout v2 Results

Date completed: 2026-07-13

## What Was Run

The protocol was frozen in [the v2 preregistration](MATCHED_HOLDOUT_V2_PREREGISTRATION.md) before any generation. I tested the existing release head from commit `52bdb7c` on 800 deterministic OpenAI SimpleQA questions and 800 deterministic Google GRANOLA Entity Questions. There was no fitting, recalibration, layer selection or threshold selection on this data.

I designed this matched follow-up after the low-coverage v1 stress test. The model was required to give one short factual answer even when uncertain. This tests whether the frozen detector transfers when new sources actually produce candidate answers. It is not a fully untouched replication and does not replace v1.

DeepSeek V4 Flash and Qwen 3.5 Flash independently graded the generations without seeing detector scores. An answer-entity ending entered the main evaluation only when both judges agreed on its final token and on `SUPPORTED` or `UNSUPPORTED`.

## Preregistered Result

The preregistered rule passed: the lower bound of the pooled prompt-bootstrap AUROC is above 0.5, and neither source AUROC is below 0.5.

| Frozen L30 correctness | N | Unsupported | AUROC | AP |
|---|---:|---:|---:|---:|
| Pooled | 1,532 | 1,105 (72.1%) | **0.844** | **0.926** |
| SimpleQA | 734 | 697 (95.0%) | **0.650** | **0.974** |
| GRANOLA | 798 | 408 (51.1%) | **0.817** | **0.821** |

The pooled prompt bootstrap used 2,000 deterministic resamples:

- AUROC mean 0.844, 95% CI **[0.821, 0.867]**.
- AP mean 0.926, 95% CI **[0.910, 0.941]**.

## Why The Pooled Number Needs Context

The pooled AUROC is not a good standalone headline. SimpleQA is almost entirely unsupported and has only 37 supported entity endings, while GRANOLA is nearly balanced. The detector also gives SimpleQA higher scores on average than GRANOLA. Comparisons across the two sources therefore push pooled AUROC above both source-specific values.

Post-hoc checks that remove or isolate those cross-source comparisons give:

- Equal-weight macro source AUROC: **0.733**.
- Within-source pair-weighted AUROC: **0.794**.
- SimpleQA-positive versus GRANOLA-negative cross-source AUROC: **0.897**.
- GRANOLA-positive versus SimpleQA-negative cross-source AUROC: **0.517**.

These checks do not replace the preregistered pooled result. They show that the balanced GRANOLA result, with 0.817 AUROC, is the cleanest evidence from this run. SimpleQA transfer is positive but weak and is estimated from very few supported cases. Its AP is only slightly above the 0.950 positive-class prevalence and should not be compared directly with GRANOLA AP.

## Coverage And Labels

| Source | Answered | Abstained | Ambiguous |
|---|---:|---:|---:|
| SimpleQA | 712 (89.0%) | 75 (9.4%) | 13 (1.6%) |
| GRANOLA | 758 (94.8%) | 37 (4.6%) | 5 (0.6%) |
| Total | 1,470 (91.9%) | 112 (7.0%) | 18 (1.1%) |

Both judges returned 1,600/1,600 valid grades. Response-status agreement was 99.06%, and no examples were excluded because a judge failed. Strict consensus retained 1,532 answer-entity endings across 1,404 prompts. The recorded API guard cost was about $0.295.

## Per-Token Stream Results

| Stream slice | Wrong-event prevalence | Product AUROC | Product AP |
|---|---:|---:|---:|
| All valid generations | 8.735% | **0.974** | **0.805** |
| Answered generations only | 10.292% | **0.970** | **0.809** |

The frozen boundary head reaches 0.987 AUROC and 0.953 AP on all generations. At the fixed product threshold of 0.5, all-stream precision is 0.899 and recall is 0.145. This threshold was not tuned on v2 and is not a deployment recommendation. The high stream metrics also benefit from short forced-answer generations and a comparatively high event prevalence. They do not establish long-form performance.

## Answer-Overlap Check

There is no exact normalized question overlap with P12/P15. Exact normalized gold-answer surface overlap occurs for 51 SimpleQA and 403 GRANOLA prompts. As preregistered, I reported this overlap but did not use it to change the main evaluation set.

A post-hoc check that removes every prompt with this answer overlap gives:

| No development-answer overlap | N | AUROC | AP |
|---|---:|---:|---:|---:|
| Pooled | 1,067 | **0.841** | **0.953** |
| SimpleQA | 688 | 0.643 | 0.977 |
| GRANOLA | 379 | 0.854 | 0.875 |

Development-answer overlap does not explain the result. The pooled no-overlap number still has the same source-composition problem described above.

## What This Establishes

V2 shows that the frozen L30 entity-completion correctness channel retains signal on new forced short-answer data without refitting. The balanced GRANOLA result is substantial and survives removal of exact development-answer overlaps. SimpleQA only shows directional transfer under severe class imbalance.

This supports the detector as a research result on matched short factual QA. It does not establish broad out-of-distribution transfer, long-form performance, a calibrated deployment threshold or production readiness.

## Packaged Evidence

`artifacts/matched_holdout_v2_evaluation.json` contains the summary. The score-blind judge files, selected prompt manifest, generations and extraction batches are retained internally but are not redistributed here pending source-dataset terms review. The summary documents the completed frozen evaluation, but this repo alone cannot independently recompute it from raw generations.
