# Evaluation of the detector

The detector was evaluated on 157,760 naturally generated tokens from Llama-3.1-8B-Instruct. Unsupported answer-entity endings make up only 1.021% of this stream, so AUROC alone can look much better than the detector behaves at a practical threshold.

I used three evaluations: prompt-grouped five-fold cross-validation, answer-entity-disjoint five-fold cross-validation and leave-one-source-out testing. Every token from one prompt stays in the same fold, and the entity-disjoint folds have zero normalized answer-key overlap.

Both detector heads are retrained inside every outer training fold. Regularization, calibration and thresholds are selected without using the outer test data. Gold entity spans define the labels but are not given to the detector at test time.

## Results

Prompt-grouped evaluation reaches 0.983 AUROC and 0.420 average precision. Entity-disjoint evaluation reaches 0.982 and 0.417. Leave-one-source-out reaches 0.980 and 0.391.

At a validation-selected threshold near one false alarm per 100 generated tokens, prompt-grouped precision is 0.368 and recall is 0.557. This is useful for a research instrument, but not enough for deployment.

The completion-aware detector improves average precision by 0.141 over the earlier entity-onset detector. A paired prompt bootstrap gives a 95% interval from +0.114 to +0.167.

The full product reaches 0.420 average precision, compared with 0.229 for entity completion alone, 0.022 for support checking without entity gating and 0.009 for token surprisal. Replacing the local support head with the published L30 probe gives 0.347.

## Reproduction

```
python scripts/verify_results.py
python scripts/verify_bootstrap.py --iterations 2000
```

This release contains the out-of-fold labels and scores, all 17 detector artifacts and their checksums.

Additional results are in the [main results](RESULTS.md) and the [entity-completion analysis](ENTITY_COMPLETION_RESULTS.md). The compact external-holdout evidence remains available as a machine-readable artifact.
