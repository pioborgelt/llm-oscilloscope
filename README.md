# LLM-Oscilloscope

> **Reviewer evidence package:** This repository exists to let grant reviewers inspect a current snapshot of my work. The detector is not ready for deployment.

The LLM-Oscilloscope is a research project about reading useful signals from the internal states of LLMs while they generate and translating them into human-readable measurements.


The long-term goal for the Oscilloscope is an EEG for model generation, which includes, but is not limited to, separate measurements for factual retrieval, answer formation and correctness.

## Why This Repo
A generous $20,000 Emergent Ventures grant supported the exploratory work that led to the current state of work. This repo contains the most important results so far in a form that reviewers can inspect directly. This includes the detector weights, OOF predictions, checksums and CPU verification scripts.

The current state required a much broader body of discovery work than what is included here. That included many failed approaches, label audits, mechanistic experiments and several different versions of the detector. The repo is just a research checkpoint for additional research support, and it does not mean hallucination detection has been solved. See [Non-Claims](#non-claims) for more information.

More detailed methods, results, limitations and information about data availability are in [`docs/`](docs/).

## What Currently Works
The detector in this repo uses two signals from the Llama-3.1-8B-Instruct LLM after each generated token. One estimates whether an answer entity has just ended, the other estimates whether that completed entity is unsupported. The product of those two signals produces a detector score on each generated token without web search, repeated sampling or a second model at inference time. However, it does require access to the model's hidden states and can only flag an entity *after* it has been emitted.


The detector was evaluated on 157,760 naturally generated tokens of which about 1% are unsupported entity endings. It produces 0.983 AUROC and 0.42 average precision on prompt-grouped folds. Leave-one-source-out evaluation produces 0.98 and 0.39.


It's important to note that the high AUROC should not be read as 98% accuracy. At an operating point near one false alarm per 100 tokens, precision is 0.368 and recall is 0.557. This is very useful research progress, but it's far from a production detector.


The exported detector was also frozen and tested without refitting on new
short-answer data. On the GRANOLA subset it reaches 0.817 AUROC.

Also see:
[per-token detector results](docs/RESULTS.md),
[entity-completion analysis](docs/ENTITY_COMPLETION_RESULTS.md) and
[matched transfer results](docs/MATCHED_HOLDOUT_V2_RESULTS.md).

## V_dyn
The current detector is only one part of the long-term intended instrument. Earlier experiments also found a separate signal at the transition from prompt processing to generation, which I called V_dyn.

While the initial interpretation was too strong, as it is not a truth detector and does not show whether the model knows it is wrong, later experiments show that it is better understood as a retrieval-timing signal. This means it can track whether the model's factual retrieval has fired, but retrieval can fire and still return the wrong person or fact, which means that V_dyn cannot replace the correctness channel.


Also, V_dyn is important for cross-model performance of this detector, as earlier experiments found that this signal can be transferred between different model architectures through linear representation alignment. The larger question is whether retrieval, entity and correctness channels can all be transferred this way without training a completely new detector.


## Relation To Existing Work
Internal-state hallucination detection already exists, and this project did not invent the concept of hidden-state probes or entity-level detection.

The closest work to this is Obeso et al.'s [Real-Time Detection of Hallucinated Entities](https://arxiv.org/abs/2509.03531).
This already demonstrates entity-level detection on long-form generations, several model families and models up to 70B parameters. [MIND](https://aclanthology.org/2024.findings-acl.854/) studies
unsupervised real-time detection from internal states, while
[Simple Factuality Probes](https://aclanthology.org/2025.findings-emnlp.880/) shows
that lightweight probes can work on long-form factuality.


Because of this, the attached two-head detector is not supposed to be the final novelty claim. It is a working proof of my work, and the part that could make the LLM-Oscilloscope meaningfully different is the combination of separate measurement channels, transfer through representation alignment and eventually using all of those signals in a closed loop to reduce errors.


## Research Roadmap
The next step is per-token grading of V_dyn. At the current state, the V_dyn signal mainly compares the state before generation with the first generated token. I want to follow the same retrieval signal through the generation sequence and test if it can distinguish the start of factual retrieval at every token position.

Combining possible per-token grading of V_dyn with the existing channels would give the detector a more complete lifecycle for factual answers, which then consists of retrieval starting, development of the retrieved content and finishing of the answer entity. All of this then receives a correctness score.

Another major test is cross-model transfer of the detector. I want to align a new model's internal space to an existing instrument and measure how much of the detector survives. Cross-model transfer is not only hypothetical: [Obeso et al.](https://arxiv.org/abs/2509.03531) report that hallucination probes can transfer between model families, and my earlier V_dyn experiments found that the retrieval signal can be moved between four architectures through linear alignment. The open question here is whether the complete multi-channel instrument transfers in the same way.

Then the detector needs to move from short factual answers to everyday long-form generation. Better independent labels and realistic calibration are part of that work.


Hallucination reduction remains in the project. Once the instrument can reliably identify where a factual answer process fails, it can be used to trigger abstention, retrieval, local regeneration or representation-level intervention. The most important test is whether these actions reduce errors without damaging correct answers. Earlier experiments already showed that a signal can be diagnostically useful without being a good steering direction, so negative results matter here.

The retrieval, entity and correctness channels are only the beginning of the intended instrument. Additional channels are on the roadmap, including which internal subspaces the model is relying on, whether an answer is being shaped by sycophancy, and whether the model is following its own retrieval signal or overriding it during generation. Together, all of these signals could show what the model is actually doing while generating.


## Non-Claims

This repository does not claim the following:

- hallucinations are detected with 98% accuracy
- the detector is production-ready or covers every type of hallucination
- natural long-form generation has already been validated
- hallucination reduction has already been achieved
- V_dyn determines whether a retrieved fact is correct
- the complete detector already transfers across model architectures

## Five-Minute Reviewer Check
The main check of the evidence:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/verify_results.py
python -m unittest discover -s tests -q
```
The verifier recomputes the evaluations and checks all 17 detector artifacts, checksums and entity-disjoint manifests.

The paired prompt bootstrap is slower and can be reproduced separately:

```bash
python scripts/verify_bootstrap.py --iterations 2000
```


## Evidence Boundary

Many of my failed approaches, activation cache data for this detector, etc. are intentionally excluded. This package shows that the reported metrics follow from the released predictions and that the packaged detector and evaluation artifacts exist.
