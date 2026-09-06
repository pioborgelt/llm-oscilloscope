# 0.6.0a0: experimental ownership assay

This preview adds `llmosci sycophancy`, a controlled comparison of how a model's
preference changes when the same position is assigned to the user. It is an
active ten-call assay, not a tokenwise probe or a detector of harmful agreement.
The existing tokenwise channels and their evidence are unchanged.

The wheel includes four GPU-free recordings: the same workshop and arithmetic
cases on SmolLM3-3B and Qwen3-1.7B. New inputs can be measured locally with
either pinned FP16 CUDA profile. Model downloads require `--allow-download`;
replay needs neither model weights nor Torch. Saved recordings are checked
and recalculated, and existing files are never overwritten.

```bash
python -m pip install ./llm_oscilloscope-0.6.0a0-py3-none-any.whl
llmosci sycophancy
llmosci sycophancy --sample qwen3-workshop
```

For a fresh local comparison, install the `sycophancy` extra and follow the
[input and runtime guide](channels/SYCOPHANCY_OWNERSHIP.md#try-it).

On 32 new curated opinion cases, mean user-assignment shifts were 21.40 pp on
Smol and 53.12 pp on Qwen3. Both use the same case population. Completed
answers followed the changing user ownership in 49/64 and 53/64 order pairs.
The underlying readout is conditional A/B preference, not a probability of
sycophancy, correctness or harm.

Important limits are part of the release:

- Neutral correctness on the 24 factual controls was only 25/48 and 23/48.
  An explicit factual question and a shared-record format did not pass the
  follow-up competence point on either model.
- Ownership wording matters. Indirectly describing the requester reduced
  Smol's effect from 21.49 to 5.54 pp on the matched development subset,
  failing its predefined robustness point.
- The same first-person wording attributed to a third party also moved
  preferences. The quoted-user effect was larger, but a raw positive score
  cannot be treated as uniquely user-specific.
- Presentation order can change the score substantially. Both orders and
  actual completed answers remain visible; invalid answers stay unknown.

The repository includes all 1,120 primary event readouts and all 1,344 new
follow-up readouts, including failed results. The four small demos are bundled
in the wheel; the complete cohort is in the source repository. No model
weights, hidden states or generated reasoning traces are redistributed.

```bash
python scripts/verify_results.py
python scripts/verify_ownership.py
python scripts/verify_ownership_controls.py
python -m unittest discover -s tests -q
```

These are internal consistency checks, not an external model reproduction.
The model runtime and CLI were also tested against already opened research
captures on the project GPU. See the [method and evidence](channels/SYCOPHANCY_OWNERSHIP.md)
and [frozen protocol](channels/SYCOPHANCY_OWNERSHIP_PROTOCOL.md).

The broader sycophancy-channel work in [issue #3](https://github.com/pioborgelt/llm-oscilloscope/issues/3)
remains open. This preview provides a narrow usable measurement, not the
originally sought reliable warning during ordinary generation.
