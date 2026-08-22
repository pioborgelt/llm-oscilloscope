# CLI research preview

> Read this document before testing the live runtime. The CLI exposes research
> scores, not deployment probabilities or a general-purpose fact checker.

The CLI makes the Oscilloscope inspectable in two ways:

- `demo` explores 28 compact Qwen2.5-7B-Instruct GPU recordings without a
  model download;
- `generate` and `replay` read four channels from a pinned
  Qwen2.5-7B-Instruct runtime after every token.

The recordings and live runtime intentionally use the same model revision,
tokenizer, layers and measurement artifacts. The public CLI was moved from
Llama to Qwen because this Qwen checkpoint is ungated and can be downloaded by
a new user without model-access approval. The original detector evidence was
collected on Llama and is still reported as Llama evidence; this CLI migration
does not rewrite that research history.

`llmosci` and `llm-oscilloscope` are identical commands. If a console script is
not on `PATH`, use `python -m llm_oscilloscope.cli` instead.

## Installation

Python 3.10 or newer is required.

For recorded samples and CPU verification only:

```bash
python -m pip install -e .
llmosci samples
```

For live generation on Linux or native Windows:

```bash
python -m pip install -e ".[runtime]"
llmosci doctor
```

The evidence-matched live profile requires an NVIDIA GPU with CUDA, compute
capability 7.5 or newer and bitsandbytes LLM.int8 support. The default 8-bit
configuration is designed to run on an 8 GB card by allowing expected CPU
offload. `--quantization none` needs substantially more memory and is a runtime
shift from the tested profile.

On Windows CMD, activate a virtual environment with:

```cmd
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -e ".[runtime]"
python -m llm_oscilloscope.cli doctor
```

`doctor` reports the platform, package availability, GPU, CUDA runtime, compute
capability, bitsandbytes version and bundled measurement artifacts. Fix any
`FAIL` before starting the model.

## Recorded GPU samples

```bash
llmosci samples
llmosci demo 1
llmosci demo biology_mitochondria
llmosci demo computer_ram --static
```

In a terminal, `demo` opens the token explorer. Use Left/Right to move one
token, Home/End to jump and `q` or Escape to close it. `--static` prints the
complete timeline, while `--json` emits the stored recording.

The archive contains 844 scored tokens from 28 Qwen generations: elementary
questions spanning eight subject areas plus natural prose. It retains original
GPU token IDs, decoded pieces, next-token probabilities, Entity Completion,
Support, their descriptive product and Subject Routing. Prompts and outputs are
unredacted. No reference answers or manual token labels are embedded.

Short factual recordings use the same post-hoc Qwen CLI candidate thresholds as
live generation: Entity `0.35` and Combined `0.3441904783`. Natural-text
recordings are score-only because they fall outside that evaluated scope;
candidate flags and coloring are disabled. These are research candidates, not
a released detector operating point.

## Live generation

The first start may download the pinned model revision:

```bash
llmosci generate \
  "Answer with only the answer: What is the capital of Australia?" \
  --allow-download
```

The runtime prints whether model downloads are allowed or only the local cache
will be used. Hugging Face handles file-download and model-loading progress.
Later cache-only starts omit `--allow-download`:

```bash
llmosci generate \
  "Answer with only the answer: What is the capital of Australia?"
```

Without `--allow-download`, a missing pinned revision is an error rather than an
implicit network request. Transformers uses the normal Hugging Face cache; the
CLI does not maintain a separate model directory.

The default `short-factual` preset uses greedy decoding and at most 32 new
tokens. This is the closest available match to the evidence, but its thresholds
are still post-hoc CLI candidates. A threshold crossing is not proof that an
answer is unsupported.

```bash
llmosci generate \
  "Explain in three short sentences why leaves look green." \
  --preset freeform \
  --max-new-tokens 96
```

`freeform` is explicitly out of distribution. Its raw scores remain visible,
but candidate thresholds and alert coloring are disabled. Sampling with a
nonzero `--temperature` is also exploratory because the available evidence uses
greedy decoding.

When generation finishes in a terminal, the token explorer opens automatically.
The latest live view and the explorer show:

- decoded token text, token ID and next-token probability;
- Entity Completion and Support scores;
- their Combined score;
- the strongest of eight Subject Routing scores.

The Support, Entity and Combined values are research scores, not calibrated
deployment probabilities. Subject Routing is a graded ranking, not a hard
router decision.

## Forced replay

```bash
llmosci replay \
  --prompt "Answer with only the answer: What is the capital of Australia?" \
  --completion "Canberra"
```

Replay tokenizes the supplied completion and forces it through the same
incremental KV-cache path. It does not claim that the model generated the
completion. Replay is score-only: candidate threshold labels and coloring are
disabled.

## Candidate threshold boundary

Short factual generation displays Entity `0.35` and Combined
`0.3441904783` as post-hoc CLI candidates. They were selected on an opened
600-prompt short factual-QA sample, not frozen before a fresh test. At the
Combined candidate, the development sample contains `TP=324`, `FP=109`,
`FN=213` and `TN=2,678` across 3,324 scorable tokens: 60.3% of unsupported
events are detected, 39.7% are missed, `FP / TP` is 0.336 and alarm precision
is 0.748 at 16.2% event prevalence.

This does not overturn the frozen `NO_SUPPORT_V2_COMBINED_ALERT_RELEASE`
decision. No threshold on that opened sample meets the older recall-at-least
40% and FPR-at-most-2% gates simultaneously, and the shuffled-support margin
also fails. A standard alert threshold still requires a fresh frozen holdout.

## JSONL traces

```bash
llmosci generate "What is the capital of Australia?" \
  --trace traces/capital.jsonl
```

The first line records the mode, prompt, exact model revision, quantization,
layers, artifact hashes, evidence boundary and whether CLI candidate thresholds
were active. Every later line is one token reading. In Replay and Freeform
traces, alert flags are disabled while the raw scores are retained. Traces
contain unredacted prompts and generated content; do not publish sensitive
inputs.

## Pinned runtime contract

The live runtime uses:

- `Qwen/Qwen2.5-7B-Instruct` at revision
  `a09a35458c702b33eeacc393d103063234e8bc28`;
- Qwen L23 post-token states for Entity Completion and Support V2;
- Qwen L27 post-token states for Subject Routing;
- 8-bit loading and greedy decoding by default;
- one captured final-position state after each emitted non-EOS token.

Only the selected final sequence position is copied to CPU. The runtime does
not request or retain every hidden state. It verifies the SHA-256 of every
bundled measurement artifact before scoring.

Use `--artifacts PATH` or `LLM_OSCILLOSCOPE_ARTIFACTS` to select a custom
evidence directory. Use `HF_TOKEN` for Hugging Face authentication when needed;
never place tokens on the command line or in trace files.
