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
- split prompt IDs inside every weight artifact
- the preregistrations, result files and cryptographic manifest

The full caches can be reproduced from the research pipeline, source datasets
and the named upstream models, subject to their access and license terms. The
complete native Llama subject-stream records were not retained in this compact
package, so only their frozen summary is included. I am also keeping the raw
matched-holdout questions, generations and judge files out of this reviewer
package until the same terms have been reviewed. Any broader data release will
need its own license review.
