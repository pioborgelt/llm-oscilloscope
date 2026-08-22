# Artifact License Review

The source code license is Apache-2.0.

The upstream cards currently list
[Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
and
[Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
under Apache-2.0. The
[MMLU](https://huggingface.co/datasets/cais/mmlu) and
[MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) dataset cards
list MIT.

Meta Llama-3.1-8B-Instruct uses the separate
[Llama 3.1 Community License](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).
This repository does not redistribute the upstream model weights or code. It
does include compact measurement heads derived from model representations. To
cover the attribution requirements that may apply to those artifacts, the
required Llama notice is included in `NOTICE` and “Built with Llama” is shown
in the README.

This review records the upstream terms checked for this release. It is not
legal advice, and downstream users remain responsible for complying with the
applicable model and dataset licenses.

The compact CLI gallery uses 23 prompts authored specifically for this research
preview and three questions from OpenAI's SimpleQA benchmark. SimpleQA is
listed under the MIT license in the official `openai/simple-evals` repository;
its copyright and permission notice are reproduced in `NOTICE`. The gallery
does not include MMLU/MMLU-Pro or GRANOLA sample text. The recording archive
distributes Qwen token IDs, numerical measurements and generated text, but not
reference answers, upstream model weights or hidden-state vectors. The separate
selection manifest includes the three SimpleQA reference answers so the
showcase categories can be audited.
