"""2. Removing the sad direction (abliteration): the model can no longer sound sad, even when asked to.

Uses the direction and layer from 01. On the test topics, asks for "two sad sentences" and "two happy sentences":
  - original model
  - direction projected out after every block while the model runs
  - control: a random direction projected out the same way
  - direction removed from the weights (embedding, every attention o_proj and MLP down_proj), a permanent edit
Damage: fact accuracy and KL divergence on 40 one-word facts.
"""
import torch, matplotlib.pyplot as plt
from common import *
from data.topics import TEST, prompts

saved = torch.load(OUT + "/sad_direction.pt")
d, LAYER = saved["direction"], saved["layer"]
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
remove_from_weights(d)
results["removed from weights"], examples["removed from weights"] = run()
for k, v in results.items():
    print("%-26s asked sad: %+.2f  asked happy: %+.2f  facts %.2f  KL %.3f | %s" % (
        k, v["asked sad"], v["asked happy"], v["facts"], v["kl"], examples[k][0][:80].replace("\n", " ")))

fig, ax = plt.subplots(figsize=(9, 4.2))
x = torch.arange(len(results))
ax.bar(x - 0.2, [v["asked sad"] for v in results.values()], 0.4, color=SERIES[1], label='asked for "two sad sentences"')
ax.bar(x + 0.2, [v["asked happy"] for v in results.values()], 0.4, color=SERIES[0], label='asked for "two happy sentences"')
ax.axhline(0, color=GRID)
ax.set_xticks(x, [k.replace(" ", "\n", 1) for k in results]); ax.set(ylabel="mood (joy - sadness)", ylim=(-1, 1),
                                                                      title=f"Remove the sad direction (layer {LAYER}) and the model can't be sad")
ax.legend(frameon=False, fontsize=9, loc="lower left")
save("02_remove_sadness", {"layer": LAYER, "results": {k: {m: r3(x) for m, x in v.items()} for k, v in results.items()},
                           "examples_asked_sad": examples}, fig)
