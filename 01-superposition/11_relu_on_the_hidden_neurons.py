"""Experiment 11. The same five concepts and two neurons, now with a ReLU on the two neurons.

Question for the viewer: the pentagon lived in a hidden layer with no ReLU on it. What happens to it
when the two hidden neurons get a ReLU of their own?

    h  = ReLU(W x)                (2 neurons)
    x' = ReLU(W^T h + b)          (5 concepts)

Same mild importance ranking as the sparsity sweep in experiment 2 (cat 1.0 ... boat 0.52).
We plot where the two neurons land for each concept on its own, and then turn the neuron axes.
Run experiment 2 first (the pentagon is loaded from its output for comparison).
"""
import json, math, torch
from common import *

IMPORTANCE = [0.85 ** i for i in range(5)]
SPARSITY = [0.0, 0.95]
LABELS = ["Always present", "Present 5% of the time"]
W, b, loss = best_of(8, SPARSITY, n=5, m=2, importance=IMPORTANCE, hidden_relu=True, steps=8000)
penta = json.load(open(os.path.join(OUT, "02_five_concepts_two_numbers.json")))["relu_S0.95"]
Wp, bp = torch.tensor(penta["W"]), torch.tensor(penta["bias"])
g = torch.Generator().manual_seed(0)
x = sample(5, torch.tensor([0.95]), 200000, g)[0]
imp = torch.tensor(IMPORTANCE)


def turned_error(W, b, deg, hidden_relu):
    """Compute the hidden layer in axes turned by `deg`, then turn back: h = R^T f(R W x)."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    R = torch.tensor([[c, -s], [s, c]])
    h = x @ W.T @ R.T
    h = (torch.relu(h) if hidden_relu else h) @ R
    return (imp * (x - torch.relu(h @ W + b)) ** 2).sum(-1).mean().item()


results = {}
fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
draw_arrows(axes[0], Wp); axes[0].set_title("No ReLU on the neurons, 5%", color=INK, pad=10)
for ax, k, label in zip(axes[1:], range(2), LABELS):
    land = torch.relu(W[k])                                   # where the two neurons land for each concept alone
    fires = [[c for i, c in enumerate(CONCEPTS) if land[j, i] > 0.3] for j in range(2)]
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.axhline(0, color=AXIS, lw=1); ax.axvline(0, color=AXIS, lw=1)
    ax.fill_between([0, 1.5], 0, 1.5, color=GRID, alpha=0.5, lw=0)
    dropped = []
    for i, name in enumerate(CONCEPTS):
        px, py = land[0, i].item(), land[1, i].item(); r = (px * px + py * py) ** 0.5
        if r < 0.15:
            dropped.append(name); continue
        ax.annotate("", xy=(px, py), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=SERIES[i], lw=2.2, mutation_scale=14))
        ax.text(px * (1 + 0.24 / r), py * (1 + 0.24 / r), name, color=INK, ha="center", va="center", fontsize=11)
    ax.text(1.45, -0.12, "neuron 1", color=MUTED, ha="right", va="top", fontsize=9)
    ax.text(-0.08, 1.45, "neuron 2", color=MUTED, ha="right", va="top", fontsize=9, rotation=90)
    ax.text(0, -1.45, "dropped: " + ", ".join(dropped), color=MUTED, ha="center", va="bottom", fontsize=9.5)
    ax.set_title("ReLU on the neurons, " + ("always" if k == 0 else "5%"), color=INK, pad=10)
    results[f"S{SPARSITY[k]}"] = dict(label=label, W=[[round(v, 3) for v in r] for r in W[k].tolist()],
                                      bias=[round(v, 3) for v in b[k].tolist()], loss=round(loss[k].item(), 5),
                                      neuron_1_fires_for=fires[0], neuron_2_fires_for=fires[1], dropped=dropped)
fig.suptitle("Where the two neurons land for each concept on its own. With a ReLU they can never go negative.", color=INK, fontsize=12.5, y=1.0)
fig.subplots_adjust(wspace=0.05)
savefig(fig, "11_relu_on_the_hidden_neurons.png")

base_p, base_r = turned_error(Wp, bp, 0, False), turned_error(W[1], b[1], 0, True)
results["turning_the_axes"] = {str(d): dict(no_relu_on_neurons=round(turned_error(Wp, bp, d, False) / base_p, 3),
                                            relu_on_neurons=round(turned_error(W[1], b[1], d, True) / base_r, 3))
                               for d in (15, 30, 45, 60, 90)}
tests = {"cat": [0], "dog": [1], "car": [2], "cat + car": [0, 2], "cat + dog": [0, 1]}
results["reading_back_at_5pct"] = {}
for name, on in tests.items():
    v = torch.zeros(1, 1, 5); v[0, 0, on] = 0.6
    out, h = forward(W[1:2], b[1:2], v, hidden_relu=True)
    results["reading_back_at_5pct"][name] = dict(neurons=[round(t, 2) for t in h[0, 0].tolist()], output=[round(t, 2) for t in out[0, 0].tolist()])
save_json("11_relu_on_the_hidden_neurons.json", results)
for k, v in results.items():
    print(k, json.dumps(v))
