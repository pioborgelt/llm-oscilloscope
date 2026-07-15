# Support Checking
This channel is the second of the two main detector channels, and it estimates whether an already completed answer entity is unsupported.

## Relation to existing work

The direct starting point for this channel is [*Real-Time Detection of Hallucinated Entities in Long-Form Generation*](https://arxiv.org/abs/2509.03531). The paper shows that hallucinated entities can be detected from a model's internal states and releases trained probe heads for several model families. This is the main prior work behind reading entity support from a post-token hidden state.

I used the paper's [official code](https://github.com/obalcells/hallucination_probes) and [published probe weights](https://huggingface.co/obalcells/hallucination-probes) as a zero-shot control on my own entities. The contribution here is not the basic idea of an internal hallucination probe. It is the cleaned support-label pipeline, the completion-aware readout and its combination with a separate entity-end channel in the naturally imbalanced token stream.


## Building the labels

Each factual span was graded by its role and support status. Because unsupported labels proved especially fragile, all 2,551 original cases were independently rejudged by Qwen 3.5 Flash and DeepSeek V4 Flash. Only agreements were kept, leaving 1,733 unsupported entities.

## The readout

Support becomes easiest to read after the complete entity has been emitted. Because of this, the channel uses the layer-30 state after the final entity token.

It reaches 0.824 AUROC and 0.681 average precision. Keeping all answer names separate between training and evaluation produces an almost identical result.

## Why it is gated

Most tokens are not completed answer entities, so support checking alone is a poor token alarm. Combining it with entity completion raises average precision from 0.229 to 0.420. Token surprisal reaches only 0.009, and the saved wrong-name example is still flagged despite a next-token confidence of 0.990.


## Frozen external checks

The frozen head was later tested without refitting. The balanced GRANOLA subset is the cleanest external result at 0.817 AUROC and 0.821 average precision. The full checks, including a weaker low-coverage test, are in the [matched external holdout](../MATCHED_HOLDOUT_V2_RESULTS.md).

## Limits

The channel checks short factual answer entities, not complete reasoning chains or every statement in long-form text. It can react after a completed entity, but it cannot warn before that entity exists.

The released weights and OOF scores make this channel independently inspectable. See [How The Detector Works](../METHOD.md), [the main results](../RESULTS.md) and [the matched external holdout](../MATCHED_HOLDOUT_V2_RESULTS.md).
