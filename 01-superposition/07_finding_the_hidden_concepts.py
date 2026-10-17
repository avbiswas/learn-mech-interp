"""Experiment 7. Can we get the hidden concepts back without being told what they are?

Question for the viewer: we hand someone ONLY the two numbers from the pentagon model, for thousands
of inputs. No labels, no weights, not even how many concepts there are. Can they find the five arrows?

We train a sparse autoencoder on the two numbers: a dictionary with more directions (8) than the
layer has dimensions (2), with a penalty that lets only a few be active at once.

Uses the model trained in experiment 2 (ReLU, each concept present 5% of the time). Run that first.
"""
import json, math, torch
from common import *

DICT = 8                                              # we deliberately do not tell it "5"
m = json.load(open(os.path.join(OUT, "02_five_concepts_two_numbers.json")))["relu_S0.95"]
W = torch.tensor(m["W"])                              # (2, 5) the true arrows. Used only to make data and to grade.
g = torch.Generator().manual_seed(0)

def activations(batch):
    x = sample(5, torch.tensor([0.95]), batch, g)[0]
    x = x[x.sum(1) > 0]                               # skip inputs with no concept in them
    return x @ W.T                                    # the two numbers. This is all the autoencoder sees.

L1 = 0.5                                              # strength of the "use few entries" penalty

def train_sae(seed, l1=L1, steps=6000):
    gs = torch.Generator().manual_seed(seed)
    D = torch.nn.functional.normalize(torch.randn(2, DICT, generator=gs), dim=0).requires_grad_()   # dictionary
    E = D.detach().clone().T.requires_grad_()         # encoder (DICT, 2)
    be = torch.zeros(DICT, requires_grad=True)
    opt = torch.optim.Adam([D, E, be], lr=3e-3)
    for _ in range(steps):
        h = activations(4096)
        Dn = torch.nn.functional.normalize(D, dim=0)
        f = torch.relu(h @ E.T + be)                  # how active each dictionary entry is
        loss = ((f @ Dn.T - h) ** 2).sum(1).mean() + l1 * f.sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        h = activations(50000)
        Dn = torch.nn.functional.normalize(D, dim=0)
        f = torch.relu(h @ E.T + be)
        return Dn, (f > 1e-3).float().mean(0), ((f @ Dn.T - h) ** 2).sum(1).mean() / (h ** 2).sum(1).mean(), (f > 1e-3).float().sum(1).mean()

runs = []
for seed, l1 in [(s, L1) for s in range(5)] + [(s, 0.05) for s in range(3)]:      # last three: penalty too weak
    Dn, freq, err, l0 = train_sae(seed, l1)
    alive = freq > 1e-3
    cosm = torch.nn.functional.normalize(W, dim=0).T @ Dn                 # (5 true, DICT learned)
    best = cosm[:, alive].max(1).values
    runs.append(dict(seed=seed, penalty=l1, alive_entries=int(alive.sum()), unexplained_fraction=round(err.item(), 4),
                     avg_entries_active_per_input=round(l0.item(), 2),
                     best_match_cosine={c: round(v, 3) for c, v in zip(CONCEPTS, best.tolist())},
                     angle_error_deg={c: round(math.degrees(math.acos(min(1, v))), 1) for c, v in zip(CONCEPTS, best.tolist())},
                     D=Dn.tolist(), alive=alive.tolist()))
    extra = [round(math.degrees(math.acos(min(1, cosm[:, j].max().item()))), 1) for j in range(DICT) if alive[j]]
    runs[-1]["each_entry_degrees_from_nearest_true_arrow"] = extra
    runs[-1]["entries_pointing_at_each_concept"] = {c: int(((cosm[i] > 0.999) & alive).sum()) for i, c in enumerate(CONCEPTS)}
    print(f"penalty {l1} seed {seed}: alive {int(alive.sum())}/{DICT}  unexplained {err.item():.3%}  active/input {l0.item():.2f}  angle error (deg)", runs[-1]["angle_error_deg"])
save_json("07_finding_the_hidden_concepts.json", runs)

r = runs[0]
Dn, alive = torch.tensor(r["D"]), torch.tensor(r["alive"])
t = math.pi / 2 - math.atan2(W[1, 0].item(), W[0, 0].item())            # same "cat points up" turn for both panels
R = torch.tensor([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
flip = torch.tensor([[-1.0], [1.0]]) if (R @ W)[0, 1] < 0 else torch.ones(2, 1)
Wr, Dr = (R @ W) * flip, (R @ Dn) * flip
h = (activations(3000) @ R.T) * flip.T
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
for ax, title in zip(axes, ["What the autoencoder is given:\njust the two numbers, as dots", "What it finds:\nits dictionary directions", "The truth:\nthe model's five arrows"]):
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axhline(0, color=GRID, lw=1); ax.axvline(0, color=GRID, lw=1); ax.set_title(title, color=INK, pad=8)
axes[0].scatter(h[:, 0], h[:, 1], s=5, color=INK2, alpha=0.5, lw=0)
for j in range(DICT):
    if alive[j]:
        hit = r["each_entry_degrees_from_nearest_true_arrow"][int(alive[:j].sum())] < 3      # lands on a true arrow?
        axes[1].annotate("", xy=(Dr[0, j].item(), Dr[1, j].item()), xytext=(0, 0),
                         arrowprops=dict(arrowstyle="-|>", color=INK if hit else MUTED, lw=2.2 if hit else 1.3, mutation_scale=13))
axes[1].text(0, -1.45, f"{int(alive.sum())} of {DICT} entries in use; they land on {len(set(round(math.degrees(math.atan2(Dr[1, j].item(), Dr[0, j].item())) / 3) for j in range(DICT) if alive[j]))} distinct directions", color=MUTED, ha="center", va="bottom", fontsize=9.5)
for i, c in enumerate(CONCEPTS):
    x, y = Wr[0, i].item(), Wr[1, i].item(); n = (x * x + y * y) ** 0.5
    axes[2].annotate("", xy=(x, y), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=SERIES[i], lw=2.2, mutation_scale=14))
    axes[2].text(x * (1 + 0.24 / n), y * (1 + 0.24 / n), c, color=INK, ha="center", va="center")
savefig(fig, "07_finding_the_hidden_concepts.png")
