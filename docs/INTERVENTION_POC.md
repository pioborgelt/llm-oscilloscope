# A First Intervention PoC
Eventually, the Oscilloscope is meant to do more than flag a problem. The long term goal is to actually intervene and trigger a more useful response. This is my first very small test of that loop, and it is a very early PoC.

## What I'm steering
`v_halluc` is an internal steering direction I found in the earlier exploratory work in this project. Despite the name I gave it earlier, it does not reliably replace a false fact with the correct one. It mainly changes how the model presents an answer, for example by moving from a confident hallucination towards hesitation.

This makes it unsuitable as an always-on intervention. However, some simple high-confidence answers survive it: for "What is 1+1?", the answer remained "2". This shows that the direction does not erase every stable answer, but the preservation result below is still too weak to call it broadly safe.

## The tested loop
First, the model generates normally while the detector reads the token stream. If no alerts fire, the answer stays untouched. If a high-confidence alert fires, the exact same prompt is generated again with `v_halluc` injected during the first five generated tokens.

The test contained 48 detector alerts from the external holdout, including 40 answers that both judges marked as incorrect. Another 24 low-score, agreed-correct prompts were kept as preservation controls.

![A saved detector alert followed by a steered abstention](../assets/funder-02-intervention.jpg)

The image shows one saved trace. The LLM first answered that Henry Stephenson was married to Winifred Shotter. The detector flagged the name, and the steered rerun changed the answer to There is not enough information to answer this question. The actual reference answer is Ann Shoemaker.

## What happened

The preregistered literal grader did not find a useful effect. It counted 2 of 40 incorrect alerts as improved under `v_halluc`, compared with 4 of 40 under the random direction. Applying the direction to every preservation prompt also retained only 6 of 24 exact reference matches.

The literal grader missed several ordinary abstention phrases, so I also report a disclosed post-hoc surface-form analysis. Under that analysis, `v_halluc` mitigates 14 of 40 incorrect alerts: 12 become abstentions and two become correct. The random direction helps 4 of 40 and produces no abstentions. This is a useful directional result, but it is not a passed preregistered reduction test.

## The evidence boundary

While this experiment shows that a detector alert can be connected to an internal intervention and that the result can sometimes be safer than the original answer, it does not show broad hallucination reduction, reliable factual repair or deployment readiness.
