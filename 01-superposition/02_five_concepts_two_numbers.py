"""Experiment 2. Five concepts, two numbers.

Question for the viewer: a model must remember how much cat, dog, car, tree and boat is in an input,
but it may only keep 2 numbers. Which concepts does it keep?

We train the same model six times: three levels of rarity, with and without the ReLU at the output.
"""
import math, torch
from common import *

IMPORTANCE = [0.85 ** i for i in range(5)]          # cat matters most, boat least
SPARSITY = [0.0, 0.8, 0.95]
LABELS = ["Always present", "Present 20% of the time", "Present 5% of the time"]

results, fig = {}, None
fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.6))
pentagon = None
for row, relu in enumerate([True, False]):
    W, b, loss = best_of(6, SPARSITY, n=5, m=2, importance=IMPORTANCE, relu_out=relu)
    if relu:
        pentagon = W[2]
    for col, S in enumerate(SPARSITY):
        ax = axes[row, col]
        dropped = draw_arrows(ax, W[col])
        if row == 0:
            ax.set_title(LABELS[col], color=INK, pad=10)
        norms = W[col].norm(dim=0)
        ang = [round(math.degrees(math.atan2(W[col][1, i].item(), W[col][0, i].item())) % 360, 1) for i in range(5)]
        results[f"{'relu' if relu else 'linear'}_S{S}"] = dict(
            kept=[c for c in CONCEPTS if c not in dropped], dropped=dropped,
            norms=[round(v, 3) for v in norms.tolist()], angles_deg=ang,
            bias=[round(v, 3) for v in b[col].tolist()], loss=round(loss[col].item(), 5),
            W=[[round(v, 4) for v in r] for r in W[col].tolist()])
    axes[row, 0].text(-0.06, 0.5, "With ReLU" if relu else "Without ReLU\n(linear)", transform=axes[row, 0].transAxes,
                      rotation=90, ha="center", va="center", color=INK2, fontsize=12)
fig.suptitle("Five concepts squeezed into two numbers. How often each concept shows up changes what the model keeps.", color=INK, fontsize=13, y=0.98)
fig.subplots_adjust(wspace=0.05, hspace=0.08)
savefig(fig, "02_five_concepts_two_numbers.png")

def record(key, W, b, loss, dropped):
    ang = [round(math.degrees(math.atan2(W[1, i].item(), W[0, i].item())) % 360, 1) for i in range(5)]
    results[key] = dict(kept=[c for c in CONCEPTS if c not in dropped], dropped=dropped,
                        norms=[round(v, 3) for v in W.norm(dim=0).tolist()], angles_deg=ang,
                        bias=[round(v, 3) for v in b.tolist()], loss=round(loss.item(), 5),
                        W=[[round(v, 4) for v in r] for r in W.tolist()])


def around_the_circle(W, b):
    """With equal importance the five concepts are interchangeable, so which one lands next to which is
    decided by the random start. Relabel them in order around the circle so every figure reads
    cat, dog, car, tree, boat. This is a renaming of the columns, not a change to the model."""
    ang = (torch.atan2(W[1], W[0]) - torch.atan2(W[1, 0], W[0, 0])) % (2 * math.pi)
    order = ang.argsort()
    return W[:, order], b[order]


# The ladder: every concept equally important, each present 5% of the time.
# Linear -> ReLU -> ReLU + bias. The first two have no bias term at all.
EQUAL = [1.0] * 5
fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.3))
ladder = [("No ReLU, no bias", "equal_linear_nobias", False, False), ("ReLU, no bias", "equal_relu_nobias", True, False),
          ("ReLU + bias", "equal_relu_bias", True, True)]
for ax, (title, key, relu, bias) in zip(axes, ladder):
    W, b, loss = best_of(8, [0.95], n=5, m=2, importance=EQUAL, relu_out=relu, bias=bias)
    W, b = (around_the_circle(W[0], b[0]) if relu else (W[0], b[0]))
    dropped = draw_arrows(ax, W, min_norm=0.0); ax.set_title(title, color=INK, pad=10)
    ax.texts[-1].set_text(f"loss {loss[0].item():.3f}")
    record(key, W, b, loss[0], [])
fig.suptitle("Five equally important concepts, each present 5% of the time, squeezed into two numbers.", color=INK, fontsize=13, y=1.0)
fig.subplots_adjust(wspace=0.05)
savefig(fig, "02b_the_ladder.png")

# The importance knob at 5% presence (ReLU + bias): equal, gently ranked, steeply ranked.
fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.3))
for ax, (title, key, decay) in zip(axes, [("All equally important", None, 1.0), ("Each 15% less important than the last", None, 0.85),
                                          ("Each half as important as the last", "steep_S0.95", 0.5)]):
    if decay == 1.0:
        m = results["equal_relu_bias"]; W, dropped = torch.tensor(m["W"]), draw_arrows(ax, torch.tensor(m["W"]))
    elif decay == 0.85:
        dropped = draw_arrows(ax, pentagon)
    else:
        W, b, loss = best_of(8, [0.95], n=5, m=2, importance=[decay ** i for i in range(5)], relu_out=True)
        dropped = draw_arrows(ax, W[0]); record(key, W[0], b[0], loss[0], dropped)
    ax.set_title(title, color=INK, pad=10)
fig.suptitle("Same rarity (5%), different importance. Cat matters most, boat least.", color=INK, fontsize=13, y=1.0)
fig.subplots_adjust(wspace=0.05)
savefig(fig, "02c_importance.png")
save_json("02_five_concepts_two_numbers.json", results)
for k, v in results.items():
    print(k, "| kept:", v["kept"], "| angles:", v["angles_deg"], "| bias:", v.get("bias", "none"), "| loss:", v["loss"])
