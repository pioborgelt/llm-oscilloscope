# Entity Correctness After Completion

This experiment tests whether entity correctness becomes easier to read after the model has emitted the complete entity. The main set contains fully observed answer-entity spans where both judges agreed. Unsupported entities are the positive class.

The protocol follows [Real-Time Detection of Hallucinated Entities in Long-Form Generation](https://arxiv.org/abs/2509.03531). I also used its [official code](https://github.com/obalcells/hallucination_probes) and [published probe weights](https://huggingface.co/obalcells/hallucination-probes) for the transfer control.

"Official zero-shot" below means that the published head was used without fitting on P15 labels. It is not fully domain-unseen because its training mixture includes TriviaQA.

## Main Result

| Model / split | First AUROC/AP | Last AUROC/AP | Span-max AUROC/AP |
|---|---:|---:|---:|
| Published head, no P15 fitting | 0.660 / 0.459 | 0.667 / 0.518 | 0.680 / 0.527 |
| Local token head, random | 0.766 / 0.578 | 0.824 / 0.681 | 0.805 / 0.663 |
| Local token head, entity-disjoint | 0.763 / 0.584 | 0.825 / 0.683 | 0.801 / 0.660 |
| Local token head, LOSO | 0.727 / 0.533 | 0.804 / 0.649 | 0.776 / 0.623 |
| Hard span max, random | 0.725 / 0.527 | 0.815 / 0.669 | 0.814 / 0.668 |
| Hard span max, entity-disjoint | 0.707 / 0.507 | 0.812 / 0.662 | 0.814 / 0.661 |
| Hard span max, LOSO | 0.687 / 0.479 | 0.799 / 0.641 | 0.796 / 0.636 |

## Paired Prompt Bootstrap

- Last - first fold-stratified AUROC: 0.059 (95% CI 0.048 to 0.070; p <0.001).
- Last - span-max fold-stratified AUROC: 0.020 (95% CI 0.012 to 0.027; p <0.001).

## All-Role Control

| Model / split | First AUROC/AP | Last AUROC/AP | Span-max AUROC/AP |
|---|---:|---:|---:|
| Official zero-shot | 0.674 / 0.250 | 0.723 / 0.347 | 0.725 / 0.348 |
| tokenwise, random | 0.852 / 0.467 | 0.883 / 0.575 | 0.872 / 0.559 |
| tokenwise, LOSO | 0.831 / 0.425 | 0.867 / 0.540 | 0.853 / 0.521 |

I do not report entity-disjoint CV for the all-role control. Repeated supporting entities connect most prompts into one large component and produce unusable fold sizes, for example 6,368 train / 8,612 test. The answer-entity-disjoint split above remains the valid primary control.

## Entity Length

| Length | N | First stratified AUROC/AP | Last stratified AUROC/AP | Span-max stratified AUROC/AP |
|---|---:|---:|---:|---:|
| one_token | 1259 | 0.804 / 0.528 | 0.804 / 0.535 | 0.804 / 0.532 |
| two_tokens | 1749 | 0.767 / 0.583 | 0.816 / 0.669 | 0.808 / 0.662 |
| three_tokens | 1200 | 0.740 / 0.585 | 0.836 / 0.715 | 0.810 / 0.693 |
| four_plus_tokens | 1215 | 0.730 / 0.611 | 0.826 / 0.739 | 0.805 / 0.721 |
| multi_token | 4164 | 0.750 / 0.591 | 0.826 / 0.707 | 0.808 / 0.689 |

The gain comes from multi-token entities. First and last are effectively identical for one-token entities, while the gap grows once the identity has to unfold over several tokens.

## What This Means

The correctness signal is substantially stronger after the full entity has been emitted. The last token beats both the first token and max aggregation over the span. Hard span-max training also does not beat the simpler token head evaluated at the last token.

The published head transfers above chance, but it loses a lot under the P15 data and labels. The local head stays almost unchanged when answer entities are disjoint between train and test, so memorized entity strings are not a good explanation for the result.

The limitation is timing: the stronger score only exists after the entity has already been generated. It helps detection, but it cannot prevent the entity from being emitted.

## Packaged Files

`artifacts/entity_completion_scores.npz` contains the OOF labels and first-, last- and max-token scores for the random, entity-disjoint and leave-one-source-out models. `artifacts/entity_completion_analysis.json` contains the reported metrics and paired prompt bootstrap. `python scripts/verify_results.py` recomputes the six AUROC/AP pairs shown for the local token heads above.
