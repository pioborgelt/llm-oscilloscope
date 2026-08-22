<p align="center">
  <img src="assets/llm-oscilloscope-header.svg" alt="LLM-Oscilloscope waveform" width="900">
</p>

# LLM-Oscilloscope

> This repository is a reviewer checkpoint and an update on my progress. It is
> not ready for production use or deployment. For context, see [Why this
> repository?](#why-this-repository)

LLM-Oscilloscope turns selected internal events during LLM generation into
human-readable measurements. Its narrow channels help inspect distinct parts
of the generation process and can be combined only where the evidence supports
that combination.

## Try the CLI

The easiest way to understand the project is to use the CLI with the included
Qwen2.5-7B-Instruct recordings. The recorded demos do not require a GPU.

```
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

llmosci samples
llmosci demo biology_mitochondria
```

`samples` lists all recordings. `demo` opens an interactive token explorer for
one sample. The archive contains 26 examples spanning Biology, Chemistry,
other academic subjects and natural text. Use the arrow keys to inspect what
each channel read after each token.

The gallery deliberately includes three detected reference-inconsistent
factual answers and three known false-positive examples. The detected cases are
the first three qualifying endpoints in a pre-existing frozen 400-prompt pool,
not the three highest scores.

Live generation is available on CUDA systems running Linux or native Windows.
The default profile is designed for an 8 GB GPU with CPU offload:

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

The public CLI uses Qwen2.5-7B-Instruct to avoid gated Llama access. The
original detector research used Llama and remains labeled accordingly below.

Please read the [CLI guide](docs/CLI.md) before testing the live runtime.

## Channels

The Oscilloscope uses narrow measurement channels for specific questions. The
current CLI exposes Entity Completion, Support Checking, Subject Routing and a
combined candidate score for unsupported entities.

### Entity completion

Entity Completion estimates whether a factual answer entity has just finished.
It reaches **0.954 AUROC / 0.824 AP** on 464 aligned endpoints from Qwen's own
generations. This determines when the Support reading is meaningful. The next
step is to extend it beyond short factual answers to longer explanations.
[Channel details](docs/entity_support_detector/ENTITY_COMPLETION.md)

### Support checking

Support Checking estimates whether a completed answer entity looks unsupported.
The supervised Qwen Support V2 head reaches **0.843 AUROC / 0.921 AP** on 655
fresh GRANOLA endpoints at 68.9% unsupported prevalence. This is a ranking
result from target-labeled onboarding, not a deployment operating point. The
next step is to test it on longer and more natural generation. [Channel
details](docs/entity_support_detector/SUPPORT_CHECKING.md)

### Combined alert

The Combined Alert multiplies Entity Completion and Support into one warning
score. It reaches **0.983 AUROC / 0.420 AP** on Llama. The Qwen CLI shows the
corresponding product of its Qwen-specific heads, but its threshold is only a
post-hoc candidate. [Method and results](docs/entity_support_detector/RESULTS.md)

### Subject routing

Subject Routing estimates which academic subject area is active during
generation. It reaches **0.933 macro AUROC on Mistral** and **0.934 on Qwen**.
It is used to test whether the same measurement can move between models. The
next step is to extend it beyond academic prompts.
[Channel details](docs/channels/SUBJECT_ROUTING.md)

### Retrieval timing

Retrieval Timing is intended to show when a model starts pulling a factual
answer from its internal knowledge. It is not part of the CLI yet. The next
step is to turn the current causal evidence into a reliable token-by-token
measurement. [Current evidence](docs/channels/RETRIEVAL_TIMING.md)

<p align="center">
  <img src="assets/cli-fixed-pool-detection.svg" alt="LLM-Oscilloscope CLI showing an unsupported entity detected in a fixed-pool Qwen trace" width="100%">
</p>

<p align="center"><em>A selected Qwen trace, where the model answers “Clare Falk” instead of Louis Kievman, and the false name's final token crosses the Combined candidate threshold.</em></p>

## Why this repository?

A $20,000 Emergent Ventures grant funded the discovery work for the initial
entity-support detector. That work included *many* failed attempts, multiple
detector versions, label audits and mechanistic experiments. I now want to turn
those probes into an instrument that can actually be used, because mechanistic
findings rarely reach the settings where they are needed.

For the next round of research and engineering, I am looking for grants,
compute and collaborators. This repository gives reviewers and other
researchers an independently inspectable record of the current work.

This reviewer repository contains the part that can be inspected independently:

- a runnable CPU CLI and optional live GPU runtime
- portable measurement heads and cross-model adapters
- out-of-fold predictions and frozen evaluation outputs
- checksums, manifests and machine-readable metadata
- verification scripts and tests

## Next research steps

For current experiments and implementation work, see the [GitHub
issues](https://github.com/pioborgelt/llm-oscilloscope/issues).

Overall, the research will further validate and optimise existing channels,
test how reliably they move between models and develop new channels for
distinct questions such as retrieval timing, sycophancy, internal routing and
retrieval override. Only channels that survive strict controls will become
part of the instrument later on.
Interventions remain a later step once a reading is reliable enough to act on.

Another point on the roadmap is reduction. Once the instrument can distinguish
where an answer process failed, different interventions can be compared under
one harness instead of being tested as isolated steering tricks.

The goal is not to add every probe ever published, but to find a small set of
readings that remain interpretable, transferable and useful for action. This
work requires resources I do not currently have, which is why I am making the
repository public.

## Current state and documentation
The reports below contain the evaluation regimes, operating points, raw counts,
controls and exact claim boundaries behind the short channel summaries above.

| Area | Current state | Start here |
|---|---|---|
| CLI | Recorded CPU demos and a live Qwen GPU research preview | [CLI guide](docs/CLI.md) |
| Completion-aware detector | Packaged Llama evidence with reproducible out-of-fold and external evaluations | [Method](docs/entity_support_detector/METHOD.md) · [Evaluation](docs/entity_support_detector/EVALUATION.md) · [Results](docs/entity_support_detector/RESULTS.md) · [Entity-completion analysis](docs/entity_support_detector/ENTITY_COMPLETION_RESULTS.md) |
| Channel notes | Entity Completion, Support Checking and Subject Routing | [Entity Completion](docs/entity_support_detector/ENTITY_COMPLETION.md) · [Support Checking](docs/entity_support_detector/SUPPORT_CHECKING.md) · [Subject Routing](docs/channels/SUBJECT_ROUTING.md) |
| Cross-model transport | Qwen-native detector-channel evidence and Subject Routing adapters | [Qwen-native transfer](docs/QWEN_NATIVE_TRANSFER.md) · [Subject Routing](docs/channels/SUBJECT_ROUTING.md) |
| Active research | Retrieval Timing and a first detector-triggered intervention PoC | [Retrieval Timing](docs/channels/RETRIEVAL_TIMING.md) · [Intervention PoC](docs/INTERVENTION_POC.md) |
| Scope and artifacts | Limitations and Non-Claims, availability and redistribution boundary | [Limitations and Non-Claims](docs/LIMITATIONS.md) · [Data availability](docs/DATA_AVAILABILITY.md) · [Artifact terms](ARTIFACT_TERMS.md) · [License review](ARTIFACT_LICENSE_REVIEW.md) |

AUROC and AP in this README are ranking metrics, not accuracy or deployment
performance. The linked evaluations report prevalence and operating-point
behavior.

## Verify the package

The compact evidence can be checked on CPU:

```
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

```
python scripts/verify_bootstrap.py --iterations 2000
```

Large activation caches are not included. The released predictions, weights
and metadata are sufficient to recompute the reported metrics and verify that
the portable artifacts load with the expected hashes. Read the [artifact
terms](ARTIFACT_TERMS.md) and [artifact license review](ARTIFACT_LICENSE_REVIEW.md)
before redistributing model-derived files.

**Research started with Llama. The public CLI runs on Qwen.**
