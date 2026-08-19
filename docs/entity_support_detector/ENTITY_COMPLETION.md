# Entity completion

The entity-completion channel estimates whether the model has just finished generating a factual answer entity. It does not decide whether that entity is correct. Its only job is finding the point where the separate support channel has a complete identity to judge.

This separation matters for names such as "Winifred Shotter." At the first token, the model has not yet revealed which person it is producing. Trying to grade that incomplete fragment loses information that becomes available once the full name has unfolded.

## Relation to existing work

The starting point for this experiment was [*Real-Time Detection of Hallucinated Entities in Long-Form Generation*](https://arxiv.org/abs/2509.03531). The paper shows that hidden-state probes can detect hallucinated entities and treats complete entity spans as the relevant unit instead of assigning one general truth score to an entire answer.

I used the paper's [official code](https://github.com/obalcells/hallucination_probes) and [published probe weights](https://huggingface.co/obalcells/hallucination-probes) as a direct transfer control. The published L30 head was applied to my P15 entities without fitting it on my labels. It transfers above chance, but performs substantially worse than a local head on this data.

This is not an exact reproduction of the paper. The paper led to the entity-span experiment, while my main question was more specific: at which token does an entity become easiest to judge, and can the detector find that point by itself during generation?

## What I tested

The P15 generations were graded for factual entities, their role in the answer and their support status. Every character span was aligned back to the generated token IDs. Across 20,017 mentions, 99.14% received exact or deterministic fallback alignment.

I then compared the support signal at the first entity token, the last entity token and the highest-scoring token anywhere inside the span. The primary set contains 5,423 fully observed answer entities on which the graders agreed.

The result is very clear for multi-token entities. Local correctness AUROC rises from 0.766 at the first token to 0.824 at the last. Across the 4,164 multi-token entities, it rises from 0.750 to 0.826. One-token entities do not show this gap because their first and last token are the same.

The last token also beats taking the maximum score anywhere inside the entity. A paired prompt bootstrap gives a last-minus-first improvement of +0.059 AUROC, with a 95% interval from +0.048 to +0.070.

## Turning this into a channel

The span experiment still knows the gold entity positions. The final detector cannot use those positions, so I trained a separate linear head to recognize answer-entity endings directly from the Llama-3.1-8B-Instruct layer-24 post-token state.

The positive target is the final token of an answer entity. At test time, the head receives only the current hidden state and outputs `P(answer_entity_end)` after every token. It reaches 0.989 AUROC and 0.859 average precision inside the naturally imbalanced stream.

That probability is multiplied by the separate support reading. This turns the paper-inspired span result into a tokenwise detector that decides for itself when the support score should matter. No gold entity position is used at test time.

## Limits

The reading only becomes available after the entity has been emitted. It can trigger a warning or regeneration, but it cannot stop the original entity before it appears.

The channel has so far been tested on short English factual-QA generations. Long-form prose, multilingual text, code and more ambiguous entity structures still need separate validation.

The exact first-versus-last results are in [Entity Correctness After Completion](../entity_support_detector/ENTITY_COMPLETION_RESULTS.md). The released scores are stored in `artifacts/entity_completion_scores.npz`, and `python scripts/verify_results.py` recomputes the packaged results.
