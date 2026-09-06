# CLI research preview

> Read this document before testing the live runtime. The CLI exposes research
> scores, and is not deployment ready.

## Installation

Python 3.10 or newer is required.

For [quick inspection](#quick-inspect) only:

```
python -m pip install -e .
llmosci samples
```

For [live generation](#live-generation) on Linux or native Windows:

```
python -m pip install -e ".[runtime]"
llmosci doctor
```

The live profile requires an NVIDIA GPU with CUDA, compute
capability 7.5 or newer and bitsandbytes LLM.int8 support. The default 8-bit
configuration is designed to run on an 8 GB card by allowing expected CPU
offload. `--quantization none` needs substantially more memory and is a runtime
shift from the tested profile.

On Windows CMD, activate a virtual environment with:

```
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -e ".[runtime]"
python -m llm_oscilloscope.cli doctor
```

`doctor` reports the platform, package availability, GPU, CUDA runtime, compute
capability, bitsandbytes version and bundled measurement artifacts. Fix any
`FAIL` before starting the model.

## Quick inspect

The new `llmosci sycophancy` command is a separate active ownership assay,
with GPU-free recorded comparisons by default and optional SmolLM3/Qwen3
live profiles. It does not add a warning to `generate` or `replay`. Its FP16
profiles do not need bitsandbytes. See the
[assay guide](channels/SYCOPHANCY_OWNERSHIP.md) for input format and limits.

```
llmosci samples
llmosci demo 1
llmosci demo biology_mitochondria
llmosci demo computer_ram --static
```

`llmosci` and `llm-oscilloscope` are identical commands. If a console script is
not on `PATH`, use `python -m llm_oscilloscope.cli` instead.

In a terminal, `demo` opens the token explorer. Use Left/Right to move one
token, Home/End to jump and `q` or Escape to close it. `--static` prints the
complete timeline, while `--json` emits the stored recording.

The archive contains 742 scored tokens from 26 Qwen generations: elementary
questions spanning eight subject areas plus natural text.



Short factual recordings use the same post-hoc Qwen CLI candidate thresholds as
live generation: Entity `0.35` and Combined `0.3441904783`. Natural-text
recordings are score-only because they fall outside that evaluated scope.

## Live generation

The first start may download the pinned model revision:

```
llmosci generate \
  "Answer with only the answer: What is the capital of Australia?" \
  --allow-download
```

For later cache-only starts, leave `--allow-download`:

```
llmosci generate \
  "Answer with only the answer: What is the capital of Australia?"
```

The default `short-factual` preset uses greedy decoding and at most 32 new
tokens. This is the closest available match to the evidence, but its thresholds
are still post-hoc CLI candidates.

```
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

- decoded token text, token ID and next-token probability
- Entity Completion and Support scores
- their Combined score
- the strongest of eight Subject Routing scores

### Forced replay

```
llmosci replay \
  --prompt "Answer with only the answer: What is the capital of Australia?" \
  --completion "Canberra"
```

Replay tokenizes the supplied completion and forces it through the same
incremental KV-cache path used by generation.


### Candidate threshold boundary

Short factual generation displays Entity `0.35` and Combined `0.3441904783` as
post-hoc CLI candidates. They were selected on an opened 600-prompt short
factual-QA development sample.



### JSONL traces

```
llmosci generate "What is the capital of Australia?" \
  --trace traces/capital.jsonl
```

The first line records the mode, prompt, model revision, quantization,
layers, artifact hashes, evidence boundary and whether CLI candidate thresholds
were active.
