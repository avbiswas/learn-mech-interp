"""Experiment 1. Two concepts, one number.

Question for the viewer: a model gets ONE number to remember both "how much cat" and "how much car".
Can it keep both?

We train the smallest possible model (2 concepts -> 1 number -> 2 concepts) on rare concepts,
once with a ReLU at the output and once without (no bias term in either), then test both on three inputs.
"""
import torch
from common import *

NAMES = ["cat", "car"]
S = [0.9]                                   # each concept present 10% of the time
tests = {"only cat (0.8)": [0.8, 0.0], "only car (0.8)": [0.0, 0.8], "cat AND car (0.8 each)": [0.8, 0.8]}
results = {}
for relu in [True, False]:
    W, b, loss = best_of(6, S, n=2, m=1, importance=[1.0, 1.0], relu_out=relu, bias=False)
    tag = "relu" if relu else "linear"
    results[tag] = dict(W=[round(v, 3) for v in W[0, 0].tolist()], bias=[round(v, 3) for v in b[0].tolist()],
                        loss=round(loss[0].item(), 5), tests={})
    for name, x in tests.items():
        out, h = forward(W, b, torch.tensor([[x]]), relu_out=relu)
        results[tag]["tests"][name] = dict(the_one_number=round(h[0, 0, 0].item(), 3),
                                           read_back=[round(v, 3) for v in out[0, 0].tolist()])
save_json("01_two_concepts_one_number.json", results)

fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), sharey=True)
for r, tag in enumerate(["relu", "linear"]):
    for c, (name, x) in enumerate(tests.items()):
        ax = axes[r, c]
        out = results[tag]["tests"][name]["read_back"]
        for i in range(2):
            ax.bar(i - 0.19, x[i], 0.34, color="none", edgecolor=INK2, lw=1.4, ls=(0, (3, 2)))
            ax.bar(i + 0.19, out[i], 0.34, color=SERIES[[0, 2][i]])
            ax.text(i + 0.19, out[i] + (0.04 if out[i] >= 0 else -0.04), f"{out[i]:.2f}", ha="center",
                    va="bottom" if out[i] >= 0 else "top", color=INK, fontsize=10)
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_xticks([0, 1]); ax.set_xticklabels(NAMES, color=INK2); ax.set_ylim(-1.0, 1.15)
        ax.grid(axis="y", lw=0.6); ax.set_axisbelow(True)
        if r == 0:
            ax.set_title(f"Input: {name}", color=INK, pad=8)
        if c == 0:
            ax.set_ylabel("With ReLU" if tag == "relu" else "Without ReLU", color=INK2, fontsize=12)
        h = results[tag]["tests"][name]["the_one_number"]
        ax.text(0.5, 0.05, f"the one number = {h:+.2f}", transform=ax.transAxes, ha="center", va="bottom", color=MUTED, fontsize=9.5)
fig.suptitle("Cat and car share one number. Dashed = what was in the input, solid = what the model reads back.",
             color=INK, fontsize=12.5, y=1.0)
fig.subplots_adjust(wspace=0.08, hspace=0.22)
savefig(fig, "01_two_concepts_one_number.png")
for tag, v in results.items():
    print(tag, "W =", v["W"], "bias =", v["bias"], "loss =", v["loss"])
    for k, t in v["tests"].items():
        print("   ", k, "->", t)
