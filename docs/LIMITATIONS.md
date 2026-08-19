# Limitations

The package in this repo has clear limits:

1. This is a research-grade detector, not a production hallucination alarm.
2. Both detector channels retain useful ranking information after transfer to Qwen2.5-7B-Instruct, but support transfer is uneven and not deployment-ready.
3. The detector identifies unsupported answer entities, not arbitrary reasoning errors or every kind of hallucination.
4. The strongest correctness score comes after the entity has already been completed.
5. At approximately one false alarm per 100 tokens, random precision/recall is 0.368/0.557.
6. Long-form generation, more model architectures, deployment calibration and reduction interventions have not been established yet.
7. The exported release head is not itself an independent test. The main detector numbers come from OOF evaluation, while the frozen holdout is reported separately.
8. The matched external holdout was designed after an earlier low-coverage stress test. Its pooled score is affected by source differences, so the balanced GRANOLA result is the cleaner number.
9. Subject routing covers eight academic subjects and short multiple-choice prompts. It has not been validated on multilingual generation, code or open-ended mixed-domain conversations.
10. Subject-transfer AUROC is much stronger than hard classification accuracy. The Mistral and Qwen heads reach only 0.625 and 0.617 accuracy on the untouched holdout.
11. Subject adapters fitted on general factual prompts miss the frozen transfer threshold. Channel onboarding still requires the right generation stage and broad operating domain.
12. Qwen-native support transfer reaches 0.717 pooled AUROC but only 0.509 on SimpleQA. Its high AP partly reflects the 80.3% unsupported prevalence.
13. The stronger Support V2 Qwen head uses target-model support labels during onboarding. Its fresh 0.843 AUROC result covers GRANOLA only, not SimpleQA or long-form generation, and its sigmoid output is not deployment-calibrated.
