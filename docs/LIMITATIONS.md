# Limitations

This package has clear limits:

- This is a research-grade detector, not a production hallucination alarm.
- The complete detector has only been tested on Llama-3.1-8B-Instruct and short factual-QA generations.
- The detector identifies unsupported answer entities, not arbitrary reasoning errors or every kind of hallucination.
- The strongest correctness score comes after the entity has been completed. It cannot warn before that entity is emitted.
- At approximately one false alarm per 100 tokens, random precision/recall is 0.368/0.557.
- Long-form generation, more model architectures, deployment calibration and reduction interventions have not been established yet.
- The calibration partition used for `release_head.npz` is not an independent test set. The main 0.983/0.420 result comes from the OOF folds; the separately frozen matched holdout is reported on its own terms.
- The matched SimpleQA/GRANOLA follow-up was designed after I diagnosed an earlier mismatched stress test. Its pooled AUROC is partly driven by the difference between sources. GRANOLA is the cleaner balanced result, while SimpleQA has only 37 supported entity ends.
- The v2 protocol was frozen locally before generation but was not externally timestamped before the run. This release documents the protocol but cannot independently prove when it was frozen.
- The matched-transfer summary is included for reviewer context, but its full generation and judge files are not redistributed pending source-dataset terms review.
