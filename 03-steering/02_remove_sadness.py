"""2. Removing the sad direction (abliteration): the model can no longer sound sad, even when asked to.

Uses the sad direction of 01 at SAD_LAYER, computed fresh (sad_directions() in common.py). On the test topics, asks for "two sad sentences" and "two happy sentences":
  - original model
  - direction projected out after every block while the model runs
  - control: a random direction projected out the same way
  - the opposite push: add -strength x the sad direction at SAD_LAYER (steer towards happy instead of deleting sad)
  - direction removed from the weights (embedding, every attention o_proj and MLP down_proj), a permanent edit
Damage: fact accuracy and KL divergence on 40 one-word facts.
"""
import torch, matplotlib.pyplot as plt
from common import *
from data.topics import TEST, prompts

LAYER = SAD_LAYER
d = sad_directions()[LAYER]
sad_req, happy_req = prompts(TEST, "sad"), prompts(TEST, "happy")
before = fact_logprobs()


def run():
    s, h = generate(sad_req), generate(happy_req)
    lp = fact_logprobs()
    return {"asked sad": avg(mood(s)), "asked happy": avg(mood(h)), "facts": facts_correct(lp), "kl": kl(before, lp)}, s[:3]


results, examples = {}, {}
results["original"], examples["original"] = run()
with remove(d):
    results["removed while running"], examples["removed while running"] = run()
with remove(random_like(d)):
    results["random direction removed"], examples["random direction removed"] = run()
for s in [1, 2, 3]:
    with add(LAYER, -s * d):
        results[f"pushed -{s}x"], examples[f"pushed -{s}x"] = run()
with add(LAYER, -2 * random_like(d)):
    results["random pushed -2x"], examples["random pushed -2x"] = run()
remove_from_weights(d)
results["removed from weights"], examples["removed from weights"] = run()
for k, v in results.items():
    print("%-26s asked sad: %+.2f  asked happy: %+.2f  facts %.2f  KL %.3f | %s" % (
        k, v["asked sad"], v["asked happy"], v["facts"], v["kl"], examples[k][0][:80].replace("\n", " ")))

fig, ax = plt.subplots(figsize=(13, 4.2))
x = torch.arange(len(results))
ax.bar(x - 0.2, [v["asked sad"] for v in results.values()], 0.4, color=SERIES[1], label='asked for "two sad sentences"')
ax.bar(x + 0.2, [v["asked happy"] for v in results.values()], 0.4, color=SERIES[0], label='asked for "two happy sentences"')
ax.axhline(0, color=GRID)
ax.set_xticks(x, [k.replace(" ", "\n", 1) for k in results]); ax.set(ylabel="mood (joy - sadness)", ylim=(-1, 1),
                                                                      title=f"Remove the sad direction (layer {LAYER}), or push against it")
ax.legend(frameon=False, fontsize=9, loc="lower left")
save("02_remove_sadness", {"layer": LAYER, "temperature": TEMPERATURE, "results": {k: {m: r3(x) for m, x in v.items()} for k, v in results.items()},
                           "examples_asked_sad": examples}, fig)
