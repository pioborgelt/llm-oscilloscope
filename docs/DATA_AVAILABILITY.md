# Data Availability

The full P15 activation cache is not part of this repo. It contains about 65 GB of hidden states from 10,017 short factual-QA generations, but it is not needed to verify the released held-out metrics.

What is included:

- every OOF label and detector score used for the reported metrics
- all fold-specific linear heads and calibrators
- a separately fitted research release head
- compact first-, last- and max-token entity-completion scores
- the machine-readable summary of the frozen matched-transfer evaluation
- portable Llama, Mistral and Qwen subject-routing heads
- native and transferred subject probabilities, random controls and the complete held-out-subject evaluation
- 7,830 Qwen post-token subject readings
- compact predictions and the portable adapter for the Qwen-native two-channel detector transfer
- the portable supervised Qwen Support V2 head, fresh GRANOLA endpoint labels,
  fixed comparison scores and twenty shuffled-label controls
- the optional trajectory-profile weights, both fresh operating-point
  prediction arrays and documentation on this channel
- 26 compact Qwen CLI recordings from project-GPU runs, containing 742
  original-token readings with all four displayed channels
- the score-blind gallery selection rule, aggregate 400-prompt pool audit and
  the three selected SimpleQA reference answers
- split prompt IDs inside every weight artifact
- the result files and cryptographic manifest
- all 112 ownership-assay recordings (56 cases on each of two models), including
  the 32 original curated opinion cases, 24 generated factual controls,
  1,120 compact event readouts, model revisions and frozen summaries
- four ownership recordings in the installable package: the same two
  preselected cases on SmolLM3 and Qwen3
- all 896 new wording-control and 256 quoted-identity readouts from the
  explicitly post-hoc follow-up tests, with their complete result summaries
- all 192 neutral factual-task diagnostic readouts, including invalid or
  incomplete answers and all four failed competence points

The full caches can be reproduced from the research pipeline, source datasets
and the named upstream models, subject to their access and license terms. The
complete native Llama subject-stream records were not retained in this compact
package, so only their frozen summary is included. I am also keeping the raw
matched-holdout questions, generations and judge files out of this reviewer
package until the same terms have been reviewed. Any broader data release will
need its own license review.

The ownership evidence contains complete account texts, alias log-sum-exp
values, A/B vocabulary mass and completed A/B answers, not full vocabulary
logits, hidden states, upstream model weights or reasoning traces. Its cases
were authored for this project; the public self-blinding paper's development
scenario texts are not redistributed here. The complete ownership cohort and
CPU verifier are in the repository; the wheel includes the four small demos.

The CLI recordings contain prompts, generated outputs, original
Qwen token IDs, decoded display pieces, next-token probabilities, channel
scores and provenance hashes, but not hidden-state vectors or manual grading.
Every token has Qwen L23 Entity Completion and Support V2, their descriptive
product and Qwen L27 Subject Routing. The package loader verifies per-recording
token-row hashes and the hash of the complete recording archive.

The gallery contains three SimpleQA questions under the upstream MIT license.
They were selected as the first three detected unsupported endpoints in a
pre-existing frozen prompt order rather than by detector score. Five earlier
sentence-style false-positive recordings were removed; the three known
Chloroplast, DNA and RAM false positives remain visible.
