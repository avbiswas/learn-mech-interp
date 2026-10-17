"""Experiment 4. How many concepts does the model keep as they get rarer?

Question for the viewer: two numbers, five concepts. We slowly make the concepts rarer.
Does the model keep more of them gradually, or does something else happen?
"""
import torch
from common import *

IMPORTANCE = [0.85 ** i for i in range(5)]
present = torch.logspace(0, -2, 33)                 # each concept present 100% ... 1% of the time
S = (1 - present).tolist()
rows = {}
for relu in [True, False]:
    W, b, loss = best_of(6, S, n=5, m=2, importance=IMPORTANCE, relu_out=relu, steps=5000)
    kept = (W.norm(dim=1) > 0.5).sum(1)             # a concept counts as kept if its arrow is long
    # error relative to a model that keeps nothing and just outputs each concept's average
    rows["relu" if relu else "linear"] = dict(kept=kept.tolist(), loss=[round(v, 5) for v in loss.tolist()])
save_json("04_how_many_concepts_fit.json", dict(present=[round(v, 4) for v in present.tolist()], **rows))

fig, ax = plt.subplots(figsize=(8.6, 4.4))
pct = (present * 100).tolist()
ax.step(pct, rows["relu"]["kept"], where="mid", color=SERIES[0], lw=2, label="with ReLU")
ax.step(pct, rows["linear"]["kept"], where="mid", color=SERIES[1], lw=2, label="without ReLU")
ax.set_xscale("log"); ax.invert_xaxis()
ax.set_xticks([100, 50, 20, 10, 5, 2, 1]); ax.set_xticklabels(["100%", "50%", "20%", "10%", "5%", "2%", "1%"])
ax.set_yticks(range(0, 6)); ax.set_ylim(0, 5.5); ax.grid(axis="y", lw=0.6); ax.set_axisbelow(True)
ax.axhline(2, color=MUTED, lw=1, ls=(0, (2, 3)))
ax.text(100, 2.12, "the model only has 2 numbers", color=MUTED, fontsize=9.5, ha="left")
ax.set_xlabel("how often each concept is present  (rarer →)"); ax.set_ylabel("concepts the model keeps (out of 5)")
ax.text(1.02, rows["relu"]["kept"][-1], "with ReLU", color=INK, va="center", ha="left", transform=ax.get_yaxis_transform())
ax.text(1.02, rows["linear"]["kept"][-1], "without ReLU", color=INK, va="center", ha="left", transform=ax.get_yaxis_transform())
ax.set_title("The rarer the concepts, the more of them fit. It happens in jumps.", color=INK, pad=10, loc="left")
savefig(fig, "04_how_many_concepts_fit.png")
for p, a, c in zip(pct, rows["relu"]["kept"], rows["linear"]["kept"]):
    print(f"present {p:6.1f}%   relu keeps {a}   linear keeps {c}")
