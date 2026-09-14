<p align="center">
  <img src="assets/llm-oscilloscope-header.svg" alt="LLM-Oscilloscope waveform" width="900">
</p>

# LLM-Oscilloscope

> This repository is a research preview for reviewers and anyone who wants to try the instrument or inspect the evidence. It is not ready for deployment.

LLM-Oscilloscope is a tool that turns parts of LLM generation into human perceivable data during generation. It uses a number of validated channels to help understanding what actually happens inside the model, and to warn when dangerous patterns become visible.

## Try the CLI

Start with the included Qwen2.5-7B-Instruct recordings. You don't need a GPU:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

llmosci samples
llmosci demo biology_mitochondria
```

`samples` lists the 26 recordings. `demo` opens one in the terminal. Use the arrow keys to move through the generated text and inspect the readings after each token.

For live generation on your own GPU, see the setup instructions in the [CLI guide](docs/CLI.md).

The CLI displays four readings from the model states:

| Reading | What it measures |
|---|---|
| Entity completion | Whether a factual answer entity has just finished |
| Support | Whether that completed entity looks unsupported |
| Combined | The product of completion and support risk |
| Subject routing | Which of eight academic subject directions is most active |

<p align="center">
  <img src="assets/cli-fixed-pool-detection.svg" alt="Recorded Qwen answer with a Combined score crossing the candidate threshold" width="100%">
</p>

<p align="center"><em>A recorded Qwen example: the model answers “Clare Falk and her husband” instead of “Louis Kievman.” The Combined score crosses a candidate threshold chosen in a post-hoc analysis.</em></p>

## Why this repository?

A $20,000 Emergent Ventures grant funded the discovery work for the initial entity support detector. Now I want to build an instrument that makes probes like this usable during generation. While we discover many interesting mechanistic findings in research, they rarely actually get used where they are needed.

I'm looking for funding, compute and people to work with on the next stage. This repository makes the current work available for review, with runnable examples, detector weights, evaluation outputs and scripts to check them.

## Research

The research started with channels trained on Llama. The CLI now uses Qwen, and part of the work is testing which measurements can move between models without training each channel again from scratch.

[Entity completion](docs/entity_support_detector/ENTITY_COMPLETION.md) estimates whether a factual entity in the answer has just ended, such as a name or place. It gates the support score so the combined detector focuses on completed entities.

[Support checking](docs/entity_support_detector/SUPPORT_CHECKING.md) estimates whether that completed entity is supported. The current Support V2 evaluation covers fresh GRANOLA data, but not SimpleQA or long-form generation.

[Combined](docs/entity_support_detector/EVALUATION.md) multiplies entity completion and support risk, forming an alert for completed entities that look unsupported. Unfortunately, the best signal for this arrives after the entity has been generated, so it can flag an answer for inspection but cannot prevent those tokens from being generated.

[Subject routing](docs/channels/SUBJECT_ROUTING.md) reads activity across eight academic subject directions, giving a graded view of which subject is most active during generation. It was evaluated on short multiple-choice prompts, where its subject rankings are more reliable than choosing a single hard label.

[Transfer from Llama to Qwen](docs/QWEN_NATIVE_TRANSFER.md) uses one adapter for completion and the original support channel, fitted without Qwen labels. Completion transfers well; support remains uneven and falls to near chance on SimpleQA.

## Verify the package

After installation, run these checks on CPU:

```bash
python scripts/verify_results.py
python -m unittest discover -s tests -q
```

The package includes predictions, weights and metadata for recomputing the reported metrics, with checksums in `MANIFEST.sha256`. Large activation caches are not included. See [data availability](docs/DATA_AVAILABILITY.md) for details and [artifact terms](ARTIFACT_TERMS.md) before redistributing model-derived files.
