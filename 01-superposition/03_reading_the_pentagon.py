"""Experiment 3. How does the pentagon read one concept back out of two numbers?

Question for the viewer: five arrows in a flat plane cannot be perpendicular. So when the input is
just "cat", the model also "sees" a bit of the other four. How does it get the right answer anyway?
And what happens when two concepts show up at once?

Uses the two pentagon models trained in experiment 2 (equal importance, each concept present 5% of the
time, with and without a bias term). Run that first.
"""
import json, torch
from common import *

STRENGTH = 0.6
saved = json.load(open(os.path.join(OUT, "02_five_concepts_two_numbers.json")))
m, m0 = saved["equal_relu_bias"], saved["equal_relu_nobias"]      # the pentagon with and without a bias term
W, b = torch.tensor(m["W"]), torch.tensor(m["bias"])
unit = torch.nn.functional.normalize(W, dim=0)
cos = unit.T @ unit
far = min(range(1, 5), key=lambda j: cos[0, j].item())         # the concept pointing most nearly opposite
cases = {"Input: cat (no bias term)": ([0], m0),
         "Input: cat": ([0], m),
         f"Input: cat + {CONCEPTS[far]}": ([0, far], m)}
results = {"strength": STRENGTH, "cosine_between_concepts": [[round(v, 2) for v in r] for r in cos.tolist()], "cases": {}}

fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), sharey=True)
for ax, (title, (on, model)) in zip(axes, cases.items()):
    W, b = torch.tensor(model["W"]), torch.tensor(model["bias"])
    x = torch.zeros(5); x[on] = STRENGTH
    raw = (W @ x) @ W                      # what each concept's arrow reads from the two numbers
    out = torch.relu(raw + b)              # after the bias and the ReLU
    results["cases"][title] = dict(true=[round(v, 2) for v in x.tolist()], raw_reading=[round(v, 2) for v in raw.tolist()],
                                   after_bias_and_relu=[round(v, 2) for v in out.tolist()])
    for i in range(5):
        ax.bar(i - 0.2, raw[i].item(), 0.36, color=SERIES[i], alpha=0.38)
        ax.bar(i + 0.2, out[i].item(), 0.36, color=SERIES[i])
        ax.plot([i - 0.42, i + 0.42], [x[i].item()] * 2, color=INK, lw=1.6, ls=(0, (3, 2)))
        ax.text(i - 0.2, raw[i].item() + (0.04 if raw[i] >= 0 else -0.04), f"{raw[i].item():+.2f}", ha="center",
                va="bottom" if raw[i] >= 0 else "top", color=INK2, fontsize=8.5)
        top = max(out[i].item(), x[i].item()) if abs(out[i].item() - x[i].item()) < 0.12 else out[i].item()
        ax.text(i + 0.2, top + 0.04, f"{out[i].item():.2f}", ha="center", va="bottom", color=INK, fontsize=8.5)
    ax.axhline(0, color=AXIS, lw=1); ax.set_xticks(range(5)); ax.set_xticklabels(CONCEPTS, color=INK2)
    ax.grid(axis="y", lw=0.6); ax.set_axisbelow(True); ax.set_title(title, color=INK, pad=8); ax.set_ylim(-1.45, 1.25)
axes[0].set_ylabel("how much of each concept")
fig.suptitle("Pale = raw reading from the two numbers.  Solid = after bias + ReLU.  Dashed line = the truth.",
             color=INK, fontsize=12, y=1.02)
fig.subplots_adjust(wspace=0.06)
savefig(fig, "03_reading_the_pentagon.png")
save_json("03_reading_the_pentagon.json", results)
print("bias:", m["bias"])
for k, v in results["cases"].items():
    print(k); print("   true  ", v["true"]); print("   raw   ", v["raw_reading"]); print("   output", v["after_bias_and_relu"])
