<p align="center">
  <img src="assets/llm-oscilloscope-header.svg" alt="LLM-Oscilloscope waveform" width="900">
</p>

# LLM-Oscilloscope

> For now, this repository is only a reviewer checkpoint package and an update of my progress. Not ready for real use cases or deployment. For more information, check out [Why this repository?](#why-this-repository)

LLM-Oscilloscope is a tool that turns parts of LLM generation into human perceivable data during generation. It uses a number of validated channels to help understanding what actually happens inside the model, and to warn when dangerous patterns become visible.



## Try the CLI
The easiest way to understand the vision is by using the CLI with the included Qwen2.5-7B-Instruct recordings, so it is possible to test without using a GPU.

```
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

llmosci samples
llmosci demo biology_mitochondria
```

`samples` lists all recorded samples. `demo` opens an interactive token explorer in a terminal for one of these samples. The included archive
contains 28 recordings from different use cases such as Biology, Chemistry and natural text. Use the arrow keys to move
through a generation and inspect what each channel read after each token.

You can also use live generation on CUDA systems running Linux or native Windows. The default profile is designed to run on an 8 GB GPU with CPU offload:

```
python -m pip install -e ".[runtime]"
llmosci doctor
llmosci generate \
  "Answer with only the answer: What is the capital of Australia?" \
  --allow-download
```

The first live run downloads Qwen2.5-7B-Instruct into the
normal Hugging Face cache. Later starts are cache-only unless
`--allow-download` is passed again.

The public CLI and its recordings now use the same pinned Qwen model. I moved
the CLI from Llama to Qwen because the Qwen checkpoint is ungated: a new user
can reproduce the runtime without first receiving model-access approval. The
original detector research was carried out on Llama and remains labeled as
such in the evidence sections below.

Please read the [CLI guide](docs/CLI.md) before testing the live runtime.

For now, the CLI is limited to reading four values from the states:

| Reading | Question |
|---|---|
| Entity completion | Has a factual answer entity just finished? |
| Support | Does that completed entity look unsupported? |
| Combined | What is the product of completion and support risk? |
| Subject routing | Which of eight academic subject directions is most active? |

Entity completion and support are components of the packaged unsupported entity detector, and their product is displayed as a combined score.

<p align="center">
  <img src="assets/cli-fixed-pool-detection.svg" alt="LLM-Oscilloscope CLI showing a reference-inconsistent Qwen answer crossing the Combined candidate" width="100%">
</p>

<p align="center"><em>A recorded Qwen example: the model answers “Clare Falk and her husband” instead of “Louis Kievman,” and the selected entity endpoint crosses the post-hoc Combined candidate.</em></p>

More channels are in active development, but not ready to be added to the CLI. For more information, see [Next research steps](#next-research-steps).

## Why this repository?

A $20,000 Emergent Ventures grant funded the discovery work for the initial entity support detector. That work included *many* failed attempts, multiple detector versions, label audits and mechanistic experiments. Now, I want to build the actual instrument to use such probes, because while we discover many interesting mechanistic findings in research, they rarely actually get used where they are needed.

For this next round of research and building, I'm looking for grants, compute and network, so this project will actually get used one day. Because of this, this repository serves as a proof of my current work and ability to execute for reviewers and other researchers.

This reviewer repository contains the part that can be inspected independently:

- a runnable CPU CLI and optional live GPU runtime
- portable measurement heads and cross-model adapters
- out-of-fold predictions and frozen evaluation outputs
- checksums, manifests and machine-readable metadata
- verification scripts and tests



## Current state

The repository packages a small number of narrow results. This section is only
an index; the linked reports contain the datasets, split design, operating
points, controls, raw counts and claim boundaries.

| Result | Headline ranking result | Evidence |
|---|---:|---|
| Completion-aware unsupported-entity detection on Llama | **0.983 AUROC / 0.420 AP** | [Method](docs/entity_support_detector/METHOD.md) · [Evaluation](docs/entity_support_detector/EVALUATION.md) · [Results](docs/entity_support_detector/RESULTS.md) |
| Balanced external GRANOLA detector check | **0.817 AUROC / 0.821 AP** | [Results](docs/entity_support_detector/RESULTS.md) · [Support checking](docs/entity_support_detector/SUPPORT_CHECKING.md) |
| Qwen-native entity completion | **0.954 AUROC / 0.824 AP** | [Qwen-native transfer](docs/QWEN_NATIVE_TRANSFER.md) |
| Qwen-native support checking | **0.717 AUROC / 0.903 AP** | [Qwen-native transfer](docs/QWEN_NATIVE_TRANSFER.md) |
| Target-labeled Qwen Support V2 on fresh GRANOLA | **0.843 AUROC / 0.921 AP** | [Support checking](docs/entity_support_detector/SUPPORT_CHECKING.md) |
| Annotation-neutral Subject Routing transfer | **0.933 Mistral / 0.934 Qwen macro AUROC** | [Subject routing](docs/channels/SUBJECT_ROUTING.md) |

AUROC and AP are ranking metrics, not accuracy or deployment performance. The
[detector evaluation](docs/entity_support_detector/EVALUATION.md) reports the
frozen operating point, prevalence and confusion counts.

Retrieval timing and intervention remain active research branches rather than
released detector channels. Their current evidence and limits are documented
in [Retrieval timing](docs/channels/RETRIEVAL_TIMING.md) and the [first
intervention PoC](docs/INTERVENTION_POC.md). The overall scope is summarized in
[Limitations](docs/LIMITATIONS.md).

## Verify the package

The compact evidence can be checked on CPU:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

python scripts/verify_results.py
python -m unittest discover -s tests -q
```

The verifier recomputes the packaged detector evaluations, checks subject
routing and Qwen transfer results, reloads the portable heads and reproduces
recorded failure gates rather than dropping them from the output.
`MANIFEST.sha256` covers the complete evidence package.

The paired prompt bootstrap is slower and can be run separately:

```bash
python scripts/verify_bootstrap.py --iterations 2000
```

Large activation caches are not included. The released predictions, weights
and metadata are sufficient to recompute the reported metrics and verify that
the portable artifacts load with the expected hashes. Read the [artifact
terms](ARTIFACT_TERMS.md) and [artifact license review](ARTIFACT_LICENSE_REVIEW.md)
before redistributing model-derived files.

## Next research steps

The immediate priority is to turn traces from the live CLI into a reproducible
validation suite, especially for Support V2 on naturally occurring and longer
text.

The wider program is to add a small number of measurements that answer distinct
questions during generation, test whether they transfer and connect useful
readings to interventions. Candidate areas include sycophancy and retrieval
override. Reduction remains a separate research branch: the existing
intervention example is a starting point, not a completed result.

## Non-claims

This repository does not claim that:

- it distributes a production-ready or general-purpose hallucination detector;
- hallucinations are detected with 98% accuracy;
- the displayed CLI scores are calibrated probabilities;
- natural long-form generation has already been validated;
- detector transfer is uniform across datasets or model families;
- retrieval firing means that the retrieved fact is correct;
- broad hallucination reduction has already been demonstrated.

The exact boundary is documented in [limitations](docs/LIMITATIONS.md) and
[data availability](docs/DATA_AVAILABILITY.md).

## Documentation

- [CLI research preview](docs/CLI.md)
- [Detector method](docs/entity_support_detector/METHOD.md)
- [Evaluation](docs/entity_support_detector/EVALUATION.md)
- [Main detector results](docs/entity_support_detector/RESULTS.md)
- [Entity-completion analysis](docs/entity_support_detector/ENTITY_COMPLETION_RESULTS.md)
- [Entity-completion channel](docs/entity_support_detector/ENTITY_COMPLETION.md)
- [Support-checking channel](docs/entity_support_detector/SUPPORT_CHECKING.md)
- [Subject-routing channel](docs/channels/SUBJECT_ROUTING.md)
- [Qwen-native detector transfer](docs/QWEN_NATIVE_TRANSFER.md)
- [Retrieval-timing research](docs/channels/RETRIEVAL_TIMING.md)
- [First intervention PoC](docs/INTERVENTION_POC.md)
- [Limitations](docs/LIMITATIONS.md)
- [Data availability](docs/DATA_AVAILABILITY.md)

**Research started with Llama. The public CLI runs on Qwen.**
