# Support Checking

This is the second part of the detector for unsupported entities. While the first part estimates whether a factual entity has ended, this estimates whether that entity is supported. The decision is post token, so it can react immediately after the entity has been created, but cannot warn beforehand.

## Methodology
The direct starting point of this channel is the [*Real-Time Detection of Hallucinated Entities in
Long-Form Generation*](https://arxiv.org/abs/2509.03531) I used its [official code](https://github.com/obalcells/hallucination_probes) and [published probes](https://huggingface.co/obalcells/hallucination-probes) as controls. The work here adds a cleaned support label pipeline and a gate for entity completion.

All 2,5k original cases were indepently judged by two LLM judges, Qwen 3.5 Flash and Deepseek V4 Flash. Only agreements of these two graders have been kept, leaving ~1.7k unsupported entities.

Support checking alone is a poor token level alarm because most tokens are not completed entities, so gating it with entity completion raises avg. precision from 0.229 to 0.420

## Standard Detector

The original readout uses the Llama layer-30 state after the final entity
token. It reaches **0.824 AUROC / 0.681 AP**; keeping all answer names disjoint
between training and evaluation gives almost the same result. The
balanced GRANOLA check reaches **0.817 AUROC / 0.821 AP**.

The annotation-neutral adapter also transfers this ranking to Qwen without
Qwen support labels. On Qwen's own generations it reaches **0.717 AUROC /
0.903 AP**, but falls to **0.509 AUROC** on SimpleQA.

## Updated detector
The released High Recall profile adds a complementary readout for uncertainty. It was trained using a Semantic Entropy probe as in [*Semantic Entropy Probes: Robust and Cheap Hallucination Detection
in LLMs*](https://arxiv.org/abs/2406.15927), and removing the part already predicted by the endpoint Support score. A final meta head combines these two endpoints, and at inference it still only needs one generation.


Across two disjoint hard factual-QA gates:

| Profile | TP | FP | FN | TN | Detected | FP / TP | Alarm precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| Standard | 895 | 67 | 314 | 190 | 74.0% | 0.0749 | 93.0% |
| High recall | 985 | 85 | 224 | 172 | 81.5% | 0.0863 | 92.1% |

TP is an unsupported endpoint correctly flagged. FP is a supported endpoint
incorrectly flagged. FN is an unsupported endpoint the detector misses. TN is
a supported endpoint correctly left unflagged.


High recall detects 7.4 percentage points more unsupported endpoints and
reduces misses from 314 to 224, a 29% reduction. False alarms rise from 67 to
85. This remains an explicit
research option rather than the default.

## Why two channels?

The high Recall profile is explicitly a research choice for settings where a
missed unsupported entity is more costly than another review. This channel is
in active development, and could potentially replace the standard support
channel if the false alarm rate can be lowered.
## Boundary and evidence

Both versions cover short factual answer entities, not complete reasoning
chains or arbitrary long-form claims. Their sigmoid scores are not
deployment-calibrated probabilities.

Also see [How The Detector Works](METHOD.md) and the
[compact Support artifacts](../../artifacts/support_v2/).
