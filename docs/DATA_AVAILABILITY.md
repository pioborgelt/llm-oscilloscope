# Data Availability

The full P15 activation cache is not part of this repo. It contains about 65 GB of hidden states from 10,017 short factual-QA generations, but it is not needed to verify the released held-out metrics.

What is included:

- every OOF label and detector score used for the reported metrics
- all fold-specific linear heads and calibrators
- a separately fitted research release head
- compact first-, last- and max-token entity-completion scores
- the machine-readable summary of the frozen matched-transfer evaluation
- split prompt IDs inside every weight artifact
- the preregistrations, result files and cryptographic manifest

The full cache can be reproduced from the research pipeline, source datasets and Meta Llama-3.1-8B-Instruct, subject to their access and license terms. I am also keeping the raw matched-holdout questions, generations and judge files out of this reviewer package until the same terms have been reviewed. Any broader data release will need its own license review.
