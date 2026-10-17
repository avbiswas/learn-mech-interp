"""Experiment 6. Growing a neuron that means two things.

Question for the viewer: now the model has real neurons (a ReLU on the hidden layer): 10 concepts,
5 neurons. When concepts are common, each neuron picks one concept. What happens to the neurons
when the concepts become rare?

    h  = ReLU(W x)            5 neurons
    x' = ReLU(W^T h + b)      10 concepts
"""
import torch
from common import *

NAMES = ["cat", "dog", "car", "tree", "boat", "piano", "pizza", "bridge", "guitar", "clock"]
IMPORTANCE = [0.75 ** i for i in range(10)]
SPARSITY = [0.0, 0.7, 0.9]
LABELS = ["Always present", "Present 30% of the time", "Present 10% of the time"]
THRESH = 0.3                                   # a neuron "fires for" a concept if its weight is above this.
# Negative weights mean the concept pushes the neuron down; they are saved in the JSON but not drawn.

RESTARTS, STEPS = 8, 8000
print("Experiment 6: 10 concepts -> 5 neurons (ReLU on the hidden layer) -> 10 concepts")
print("concepts:  ", ", ".join(NAMES))
print("importance:", ", ".join(f"{v:.2f}" for v in IMPORTANCE))
print(f"training {len(SPARSITY)} datasets x {RESTARTS} random starts = {len(SPARSITY) * RESTARTS} models, {STEPS} steps each\n")
print(f"{'step':>6} | " + " | ".join(f"{l:^24}" for l in LABELS))
print(f"{'':>6} | " + " | ".join(f"{'loss':>9}  {'polysemantic':>12}" for _ in LABELS))


def show_progress(step, loss, Wall):
    """Best of the random starts for each dataset: its loss, and how many of its 5
    neurons fire for 2+ concepts right now."""
    k = len(SPARSITY)
    L = loss.view(RESTARTS, k)
    cells = []
    for d in range(k):
        r = L[:, d].argmin().item()
        Wd = Wall[r * k + d]
        poly = int(((Wd > THRESH).sum(1) > 1).sum().item())
        cells.append(f"{L[r, d].item():>9.4f}  {poly:>6} of 5 ")
    print(f"{step:>6} | " + " | ".join(cells), flush=True)


W, b, loss = best_of(RESTARTS, SPARSITY, n=10, m=5, importance=IMPORTANCE, hidden_relu=True, steps=STEPS,
                     log=show_progress, log_every=500)
print()
results = {}
fig, axes = plt.subplots(3, 1, figsize=(8.6, 9.2), sharex=True)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ_BLUE)
for k, (S, ax) in enumerate(zip(SPARSITY, axes)):
    Wk = W[k]
    order = sorted(range(5), key=lambda j: Wk[j].argmax().item())     # sort neurons by their main concept
    Wk = Wk[order]
    cards, n_poly = [], 0
    for j in range(5):
        hits = [(NAMES[i], round(Wk[j, i].item(), 2)) for i in Wk[j].argsort(descending=True).tolist() if Wk[j, i].item() > THRESH]
        n_poly += len(hits) > 1
        cards.append(hits)
    stored = [NAMES[i] for i in range(10) if Wk[:, i].norm().item() > 0.5]
    results[f"S{S}"] = dict(label=LABELS[k], neurons=cards, neurons_with_more_than_one_concept=n_poly,
                            concepts_represented=stored, loss=round(loss[k].item(), 5),
                            W=[[round(v, 3) for v in r] for r in Wk.tolist()], bias=[round(v, 3) for v in b[k].tolist()])
    # Drawn the same way round as W in the script: one row per neuron, one column per concept.
    ax.imshow(Wk.clamp(min=0).numpy(), cmap=cmap, vmin=0, vmax=1.2, aspect="auto")
    ax.set_yticks(range(5)); ax.set_yticklabels([f"neuron {j+1}" for j in range(5)], color=INK2)
    ax.set_xticks(range(10)); ax.set_xticklabels(NAMES, color=INK2)
    ax.set_yticks([x - 0.5 for x in range(6)], minor=True); ax.set_xticks([y - 0.5 for y in range(11)], minor=True)
    ax.grid(which="minor", color=SURFACE, lw=2); ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"{LABELS[k]}: {n_poly} of 5 neurons fire for 2+ concepts", color=INK, pad=8, fontsize=11)
fig.suptitle("10 concepts, 5 neurons. Each row is one neuron; bright = the neuron fires for that concept.",
             color=INK, fontsize=12, y=0.95)
fig.subplots_adjust(hspace=0.28)
savefig(fig, "06_neurons_that_mean_two_things.png")
save_json("06_neurons_that_mean_two_things.json", results)
for k, v in results.items():
    print(f"=== {v['label']}  (final loss {v['loss']}, {v['neurons_with_more_than_one_concept']} of 5 neurons polysemantic)")
    print("          " + " ".join(f"{n:>6}" for n in NAMES))
    for j, row in enumerate(v["W"]):
        print(f"  neuron {j + 1} " + " ".join(f"{x:>6.2f}" for x in row))
    for j, c in enumerate(v["neurons"]):
        print(f"    neuron {j + 1} fires for:", ", ".join(f"{n} ({w})" for n, w in c) or "nothing")
    print()
