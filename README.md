# LLM-Oscilloscope
> This repo is a checkpoint of my research progress for reviewers and potential funders. It contains a working but limited detector, and it is not ready to be deployed.


Because of the black-box problem, LLMs expose very little about what they are actually doing while they generate. Token probabilities can show that a model is confident, but they cannot show whether factual retrieval happened, which internal knowledge region was used, or whether an answer is actually supported.

Because of this, I'm building the LLM-Oscilloscope as an EEG for model generation. Its purpose is to translate some of the complex features in the model's internal representations into human-perceivable data.


![The four working measurement channels and two planned additions](assets/funder-04-channel-map.jpg)
The long-term goal is a model-independent research instrument that can show where an answer process starts to fail and use that information to trigger a targeted intervention.


## Why this repo?

A generous $20,000 Emergent Ventures grant funded the broad discovery work that led the detector to this point. This work included *many* failed approaches, label audits, mechanistic experiments and several versions of the detector. Most of this discovery work is intentionally not included here.

What is included is the part that can already be inspected cleanly by reviewers. This includes a portable detector head, OOF predictions, frozen evaluations, checksums, preregistrations and CPU verification scripts.


This is mainly here to prove that the current detector really exists and to provide a starting point for the next stage of the project.

## The current detector
The most complete part of the Oscilloscope is a per-token detector for unsupported (hallucinated) answer entities in Llama-3.1-8B-Instruct.

It uses two internal readings after every generated token. The first estimates whether a factual answer entity has just finished, and the second estimates whether that completed entity looks unsupported. Multiplying the two readings produces one alert score for every token in the natural generation stream.

Unlike many other methods, this does not require web search, repeated sampling or a second LLM at inference time. However, it does require access to the model's internal states, and it can only judge an entity *after* it has been emitted.


![A saved generation where the detector marks the wrong person name as unsupported](assets/funder-01-detection.jpg)
The example in the picture above is useful because the model is not uncertain in the normal sense. The model gives us a wrong name with a next-token probability of 0.990, and my detector is reading something different from ordinary output confidence.


The first channel waits until the full name has unfolded. This is especially important for multi-token entities, because correctness is substantially easier to read at the last entity token than at the first. Then the second channel checks the completed identity instead of trying to decide whether an incomplete fragment is correct.

Only entity completion and support checking feed the released alert. The retrieval reading shown in the trace belongs to the wider research instrument described below.


The detailed methodology and results are in the [detector method](docs/METHOD.md), [entity-completion analysis](docs/ENTITY_COMPLETION_RESULTS.md) and [main results](docs/RESULTS.md).


## Evaluation

The detector was evaluated on 157,760 naturally generated tokens. Unsupported answer-entity endings made up 1.021% of that stream.


On prompt-grouped out-of-fold evaluation it reaches 0.983 AUROC and 0.420 average precision. Leave-one-source-out evaluation reaches 0.980 AUROC and 0.391 average precision.

However, the high AUROC does not mean 98% accuracy. At an operating point near one false alarm per 100 generated tokens, precision is 0.368 and recall is 0.557. That is enough to make the detector useful as a research instrument, but not enough for deployment.


The exact evaluation arrays and detector heads are included here. `python scripts/verify_results.py` recomputes the reported metrics and reloads all 17 exported artifacts. The full protocol, controls and external checks are in [How I Evaluated The Detector](docs/EVALUATION.md).

## The wider instrument
The released detector contains entity completion and support checking. The wider project already has two additional working research channels. Each channel now has its own methodology and evidence boundary under [`docs/channels/`](docs/channels/).

**[Subject routing](docs/channels/SUBJECT_ROUTING.md)** reads which academic subject region the model is using while it generates. On Llama-3.1-8B it reaches 0.924 token-level macro AUROC across eight subjects. More importantly, this channel can be moved into Mistral-7B and Qwen2.5-7B through compact activation adapters while the original Llama measurement head remains frozen.

**[Retrieval timing](docs/channels/RETRIEVAL_TIMING.md)** tracks whether the model's factual retrieval process fires around the transition into generation. It is not a truth detector because retrieval can fire and still return the wrong fact. Its value is that it measures a different stage of the answer process from entity completion and support.

**[Entity completion](docs/channels/ENTITY_COMPLETION.md)** estimates when the model has finished producing a factual answer entity. This is the first half of the released detector.

**[Support checking](docs/channels/SUPPORT_CHECKING.md)** estimates whether that completed answer looks unsupported. This is the second half of the released detector.

Two additional channels are currently on the research roadmap. One would track whether an answer is being bent toward the user's stated belief or preference. The other would track whether the model follows its own retrieval signal or overrides it later in generation.

Each channel answers a narrower question than a general confidence score. The main research bet is that reading them together will be more useful than expecting one probe to explain the entire generation.

## Moving the instrument between models

Training every channel again from new labels for every model would make the Oscilloscope much less useful. A central part of the project is therefore fast channel transfer through representation alignment.

![Internal representation spaces being aligned so one measurement can be read across model families](assets/funder-03-transfer.jpg)

For the subject-routing channel, paired activations from the new model are aligned with the existing Llama instrument using a 64-dimensional PCA and Ridge adapter. No target-model subject labels are used either to train or to select the adapter.

On an untouched holdout of 128 new base questions rendered through four templates, the frozen channel achieves macro AUROCs of 0.933 on Mistral-7B and 0.934 on Qwen2.5-7B. The final adapters are smaller than 1 MB and can be fitted in roughly one CPU second once activations have been extracted.

Performance is weaker on the more demanding independent factual-prompt control, reaching AUROCs of 0.739 and 0.746, respectively, and narrowly missing the frozen 0.750 pass threshold. For now, these results should therefore be interpreted as evidence of functional channel transfer when domain-covering paired activations are available, not as evidence for a universal mapping between complete model spaces.

This is still in active testing and will be one of the first parts of the research roadmap.

## Closing the loop
But detection is only the first step. The LLM-Oscilloscope is ultimately meant to improve what the model produces by connecting its measurements to interventions at the latent-space level.


![A detector alert removing an unsupported person name during a gated rerun](assets/funder-02-intervention.jpg)


The picture above shows a basic loop for a first intervention PoC. The detector flags the wrong name, and in the second pass the model is nudged toward saying "I don't know" instead of repeating it.

But this is just one intervention example, not a hallucination-reduction result. The system did not recover the correct spouse and broad reduction has not yet been evaluated. I still include it here because it proves that the detector output can already be connected to an action, while keeping the actual reduction claim open for the next stage of research. The frozen test and its limits are described in [A First Intervention PoC](docs/INTERVENTION_POC.md).

## The roadmap

The first priority is transferring the complete instrument. Entity completion and support checking need to be moved to at least one new architecture with the same frozen-head protocol used for subject routing. This would show whether the transfer result is a reusable onboarding method or only works for one easier channel.

The second priority is moving from short factual QA to natural long-form generation. That requires better entity coverage, independent labels, realistic calibration and enough context to follow several factual claims in one answer.

The third priority is reduction. Once the instrument can distinguish where an answer process failed, different interventions can be compared under one harness instead of being tested as isolated steering tricks.

In parallel, the Oscilloscope can gain additional channels for behavior such as sycophancy, internal routing and retrieval override. The goal is not to add every probe ever published. The goal is to find a small set of readings that remain interpretable, transferable and useful for action.

However, all of this requires substantial resources that I don't currently have, which is why I'm making this repo.

## Non-Claims

This repository does not claim that:

- hallucinations are detected with 98% accuracy
- the detector is production-ready or covers every type of hallucination
- natural long-form generation has already been validated
- hallucination reduction has already been demonstrated broadly
- retrieval firing means that the retrieved fact is correct
- the complete detector already transfers across model architectures
- the released artifacts prove every wider channel described above

The exact evidence boundary is described in [limitations](docs/LIMITATIONS.md) and [data availability](docs/DATA_AVAILABILITY.md).

## Quick reviewer check

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/verify_results.py
python -m unittest discover -s tests -q
```

The verifier recomputes the three main per-token evaluations, checks the entity-completion results and reloads every exported detector artifact. `MANIFEST.sha256` covers the complete evidence package.

The paired prompt bootstrap is slower and can be reproduced separately:

```bash
python scripts/verify_bootstrap.py --iterations 2000
```

The 65 GB activation cache is not included. The released predictions and weights are enough to verify the reported metrics and confirm that the portable heads reproduce their saved outputs. See [artifact terms](ARTIFACT_TERMS.md) and the [artifact license review](ARTIFACT_LICENSE_REVIEW.md) before redistributing model-derived files.

## Documentation

The detailed technical material lives in [`docs/`](docs/):

- [Method](docs/METHOD.md)
- [Evaluation](docs/EVALUATION.md)
- [Main detector results](docs/RESULTS.md)
- [Entity-completion analysis](docs/ENTITY_COMPLETION_RESULTS.md)
- [First intervention PoC](docs/INTERVENTION_POC.md)
- [Entity-completion channel](docs/channels/ENTITY_COMPLETION.md)
- [Support-checking channel](docs/channels/SUPPORT_CHECKING.md)
- [Subject-routing channel](docs/channels/SUBJECT_ROUTING.md)
- [Retrieval-timing channel](docs/channels/RETRIEVAL_TIMING.md)
- [Matched external holdout](docs/MATCHED_HOLDOUT_V2_RESULTS.md)
- [Preregistration](docs/PREREGISTRATION.md)
- [Limitations](docs/LIMITATIONS.md)
- [Data availability](docs/DATA_AVAILABILITY.md)
