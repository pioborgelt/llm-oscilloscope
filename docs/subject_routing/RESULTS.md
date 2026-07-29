# Subject Routing Results
Subject routing is a channel that reads a graded distribution over eight academic
subjects from one hidden state.
It is separate from the existing unsupported entity alert and does not change the detector's score.


## Native channel

The primary Llama test transfers between MMLU and the non-MMLU part of MMLU-Pro. With this, the pooled cross-source result is 0.878 macro AUROC.
Held-out templates reach 0.957 and held-out fine subjects reach 0.877.

The frozen Llama summary contains 7,927 post-token states and reaches a macro AUROC of 0.924. The native stream records are not retained in this repo for size reasons, so this number is recorded but not independently recomputable here.

The lexical control mean directional source-transfer AUROC is 0.852, compared with 0.908 for the learned internal channel. This means that the prompt text already contains much of the subject information. The internal
readout matters because it can be measured after every token and transported with the model's hidden state.

## Transferring the channel
The frozen Llama channel was moved into Mistral-7B and Qwen2.5-7B using paired activations without target-model subject labels.
On 128 base questions phrased using four templates, Mistral reaches 0.933 macro AUROC and Qwen reaches 0.934. Randomly paired controls reach 0.499 and 0.488.

The included Qwen stream records contain 7,830 post-token states from 256 generations. The transported channel reaches 0.885 token-level macro AUROC
and 0.953 after averaging each generation.

## Harder tests
First adapters saw paired examples from all eight subjects, so in a second frozen test both subjects were removed from Qwen preprocessing and adapter fitting for every possible pair. The unchanged Llama channel reaches 0.953 mean AUROC across all 28 held-out pairs.


A broader control does not pass. Adapters fitted on 1,000 general factual-QA prompts reach 0.739 on Mistral and 0.746 on Qwen. Both miss the originally frozen 0.750 threshold and are released as failures. This is a current boundary, the onboarding prompts can omit individual subject labels, but they still need to cover the channel's broader domain and generation stage.

## This proves
The released evidence supports one portable subject measurement across three open model families.
The verifier recomputes the native and transported metrics, reloads the three portable heads, checks both failed controls and reproduces the complete decision. Run it with:

```
python scripts/verify_subject_routing.py
```

The same checks are also part of `python scripts/verify_results.py`.
