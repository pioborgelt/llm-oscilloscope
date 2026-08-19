# How The Detector Works

After every generated token, two regularized linear heads read hidden states from Llama-3.1-8B-Instruct:

1. The L24 post-token head estimates whether an answer entity just ended.
2. The L30 post-token head estimates whether that entity is unsupported.

The detector score is the product of those two probabilities. This keeps entity detection and correctness separate: the first head finds a candidate entity ending, and the second estimates whether the completed entity is unsupported. Both heads produce a value at every token, but the correctness value alone is not treated as a general truth score.

The held-out evaluations retrain both heads inside each outer training fold. Model fitting, regularization selection, calibration and threshold selection all stay inside that fold. The detector never receives gold entity positions at test time.

A positive event is the final token of a strict-consensus `ANSWER + UNSUPPORTED` entity. Other scorable tokens are negative, while disputed or unknown answer-entity endings are ignored.

I evaluated the detector with prompt-grouped five-fold CV, answer-entity-disjoint five-fold CV and leave-one-source-out testing. A separate control compares the L30 correctness head at the first, last and highest-scoring token of an entity. The matched holdout uses the already exported release head without fitting or recalibration.

The exact numbers are in the [main results](RESULTS.md) and the [entity-completion results](ENTITY_COMPLETION_RESULTS.md). The compact matched-holdout evidence remains available as a machine-readable artifact.
