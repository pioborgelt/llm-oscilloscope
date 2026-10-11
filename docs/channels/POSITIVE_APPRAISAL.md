# Positive appraisal

This experimental channel estimates how strongly an answer expresses praise
or approval. A small probe reads the model's internal activity after each
generated token.

It is one of RuxyTrace's first channels toward **sycophancy detection**.
I plan to add more channels that examine other aspects of this behavior.
Praise can be appropriate: this channel does not check whether it is deserved
or whether the model is agreeing with a false claim.

**Reading the score.** Higher values suggest stronger praise. The CLI shows
an average over four tokens, with green, yellow and red indicating increasing
signal strength. These colors are not safety judgments, and the score is not
a probability.

**Try it.** Open a recorded example without downloading a model:

```bash
llmosci demo appraisal_native_bike
```

For live use, add `--appraisal` to `generate` or `replay`. See the
[CLI guide](../CLI.md#positive-appraisal) for setup and more examples.

**What has been tested.** The probe was trained on AI-rated Qwen3 answers and
transferred to Qwen2.5-7B. The development checks replayed those answers through
Qwen2.5; they do not establish reliability on new conversations. A separate
attempt to turn the score into a yes/no alarm failed. Evidence is mainly in
English. The recorded examples illustrate the channel and are not a
performance test.

[Training and evaluation details](../../artifacts/appraisal_experimental/manifest.json).

The documentation for this channel will be expanded in a future release.
