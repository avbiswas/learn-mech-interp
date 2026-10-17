"""Experiment 14. Does superposition arrive gradually or all at once?

Question for the viewer: when we watch the loss during training, does the network slide into superposition
or switch into it?

Part A: two concepts, one number (experiment 1). A linear network and a ReLU network start from the same
        random weights and are trained slowly, so the curves can be compared step by step.
Part B: five concepts, two numbers (experiment 2, equal importance, 5% presence, ReLU, no bias). Trained
        slowly from 10 different random starts. We record the loss and how the five arrows are arranged.

"Slowly" matters. Experiments 1 and 2 use a fast optimiser and the whole thing is over in about 200 steps.
Here the step size is small so that any plateau is visible. The end result is the same.
"""
import json, math, torch
from common import *


def train_slow(n, m, S, seed, relu=True, steps=8000, lr=0.05, adam=False, every=25):
    g = torch.Generator().manual_seed(seed)
    W = (torch.randn(1, m, n, generator=g) * 0.3).requires_grad_(); b = torch.zeros(1, n)
    opt = torch.optim.Adam([W], lr=lr) if adam else torch.optim.SGD([W], lr=lr)
    xe = sample(n, torch.tensor([S]), 100000, g); hist = []
    for t in range(steps + 1):
        if t % every == 0:
            with torch.no_grad():
                out, _ = forward(W, b, xe, relu_out=relu)
                hist.append((t, ((xe - out) ** 2).sum(-1).mean().item(), W[0].clone()))
        x = sample(n, torch.tensor([S]), 1024, g); out, _ = forward(W, b, x, relu_out=relu)
        loss = ((x - out) ** 2).sum(-1).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return hist


def directions(w, tol=20):
    """How many distinct directions the stored arrows point in (arrows within `tol` degrees count as one)."""
    ang = sorted(math.degrees(math.atan2(w[1, i].item(), w[0, i].item())) % 360 for i in range(w.shape[1]) if w[:, i].norm() > 0.5)
    if not ang:
        return 0, 0
    groups = 1 + sum(1 for i in range(1, len(ang)) if ang[i] - ang[i - 1] > tol)
    if len(ang) > 1 and (ang[0] + 360 - ang[-1]) <= tol and groups > 1:
        groups -= 1
    return len(ang), groups


results = {}

# ---- Part A: two concepts, one number ----
SEED_A = 0
A = {"relu": train_slow(2, 1, 0.9, SEED_A, relu=True, steps=3000), "linear": train_slow(2, 1, 0.9, SEED_A, relu=False, steps=3000)}
split = next(t for (t, lr_, _), (_, ll, _) in zip(A["relu"], A["linear"]) if lr_ < 0.8 * ll)       # first step where ReLU is 20% below linear
results["two_concepts_one_number"] = dict(
    seed=SEED_A, start_weights=[round(v, 2) for v in A["relu"][0][2].flatten().tolist()],
    final_loss={k: round(v[-1][1], 4) for k, v in A.items()}, final_weights={k: [round(x, 2) for x in v[-1][2].flatten().tolist()] for k, v in A.items()},
    curves_separate_at_step=split, loss_at_separation={k: round(next(l for t, l, _ in v if t == split), 4) for k, v in A.items()},
    all_seeds=[])
for seed in range(8):
    h = train_slow(2, 1, 0.9, seed, relu=True, steps=3000); w = h[-1][2].flatten()
    results["two_concepts_one_number"]["all_seeds"].append(dict(seed=seed, final_loss=round(h[-1][1], 4), opposite_signs=bool(w[0] * w[1] < 0)))

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
ax = axes[0]
for k, c, lab in [("linear", SERIES[1], "No ReLU"), ("relu", SERIES[0], "ReLU")]:
    ax.plot([r[0] for r in A[k]], [r[1] for r in A[k]], color=c, lw=2.2, label=lab)
    ax.text(3030, A[k][-1][1], f"{A[k][-1][1]:.3f}", color=INK, va="center", fontsize=10)
ax.axvline(split, color=MUTED, lw=1, ls=(0, (3, 3))); ax.text(split + 40, 0.05, "the curves separate", color=INK2, fontsize=9.5, va="top")
ax.set_yscale("log"); ax.set_xlim(0, 3350); ax.set_xlabel("training step"); ax.set_ylabel("loss (log scale)"); ax.legend(frameon=False, loc="lower left")
ax.grid(axis="y", lw=0.6); ax.set_title("Same starting weights, two networks", color=INK, pad=8)
ax = axes[1]
for i, (name, c) in enumerate([("cat", SERIES[0]), ("car", SERIES[2])]):
    ax.plot([r[0] for r in A["relu"]], [r[2][0, i].item() for r in A["relu"]], color=c, lw=2.2, label=f"{name} weight, ReLU")
    ax.plot([r[0] for r in A["linear"]], [r[2][0, i].item() for r in A["linear"]], color=c, lw=1.6, ls=(0, (4, 3)), label=f"{name} weight, no ReLU")
ax.axhline(0, color=AXIS, lw=1); ax.set_xlabel("training step"); ax.set_ylabel("weight"); ax.legend(frameon=False, fontsize=9, ncol=2, loc="lower right")
ax.grid(axis="y", lw=0.6); ax.set_title("The two weights during training", color=INK, pad=8)
fig.subplots_adjust(wspace=0.24)
savefig(fig, "14a_two_concepts_loss.png")

# ---- Part B: five concepts, two numbers ----
STEPS_B, runs = 12000, {}
for seed in range(10):
    h = train_slow(5, 2, 0.95, seed, relu=True, steps=STEPS_B, lr=3e-4, adam=True, every=50)
    final = h[-1][1]
    stuck = [(t, l) for t, l, w in h if directions(w) == (5, 4)]            # five arrows stored but only four directions
    if stuck:                                                               # keep only the flat part, not the descent into it
        level = sorted(l for _, l in stuck)[len(stuck) // 2]
        stuck = [(t, l) for t, l in stuck if abs(l - level) < 0.02 * level]
    first5 = next((t for t, l, w in h if directions(w) == (5, 5) and l < 1.03 * final), None)
    runs[seed] = dict(hist=h, final=final, plateau_steps=(stuck[0][0], stuck[-1][0]) if len(stuck) >= 10 else None,
                      plateau_loss=sum(l for _, l in stuck) / len(stuck) if len(stuck) >= 10 else None, pentagon_from=first5)
with_plateau = [s for s, r in runs.items() if r["plateau_steps"]]
results["five_concepts_two_numbers"] = dict(
    seeds=10, runs_that_end_in_a_pentagon=sum(1 for r in runs.values() if directions(r["hist"][-1][2]) == (5, 5)),
    runs_with_a_four_direction_plateau=len(with_plateau),
    per_seed={str(s): dict(final_loss=round(r["final"], 4), plateau_steps=r["plateau_steps"],
                           plateau_loss=round(r["plateau_loss"], 4) if r["plateau_loss"] else None, pentagon_from_step=r["pentagon_from"]) for s, r in runs.items()})
show = max(with_plateau, key=lambda s: runs[s]["plateau_steps"][1] - runs[s]["plateau_steps"][0]) if with_plateau else 0
R = runs[show]; results["five_concepts_two_numbers"]["shown_seed"] = show
if R["plateau_steps"]:
    results["five_concepts_two_numbers"]["drop_at_the_jump"] = f"{R['plateau_loss']:.4f} -> {R['final']:.4f} ({100 * (1 - R['final'] / R['plateau_loss']):.0f}% lower)"

fig = plt.figure(figsize=(11.5, 7.2)); gs_ = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.3, wspace=0.05)
ax = fig.add_subplot(gs_[0, :])
for s, r in runs.items():
    if s != show:
        ax.plot([x[0] for x in r["hist"]], [x[1] for x in r["hist"]], color=MUTED, lw=1, alpha=0.55)
ax.plot([x[0] for x in R["hist"]], [x[1] for x in R["hist"]], color=SERIES[0], lw=2.4)
ax.set_ylim(R["final"] * 0.97, R["final"] * 1.45); ax.set_xlim(0, STEPS_B); ax.set_xlabel("training step"); ax.set_ylabel("loss"); ax.grid(axis="y", lw=0.6)
picks = [("early", 1000)]
if R["plateau_steps"]:
    p0, p1 = R["plateau_steps"]; picks.append(("on the plateau", (p0 + p1) // 2))
    ax.axvspan(p0, p1, color=GRID, alpha=0.9, lw=0); ax.text((p0 + p1) / 2, R["plateau_loss"] * 1.035, "plateau: five arrows, four directions", color=INK2, ha="center", fontsize=9.5)
picks.append(("end", STEPS_B))
ax.set_title(f"Loss during training. Blue: one run. Grey: the other {len(runs) - 1} runs.", color=INK, pad=8)
for col, (label, step) in enumerate(picks):
    a2 = fig.add_subplot(gs_[1, col]); w = next(w for t, l, w in R["hist"] if t >= step)
    draw_arrows(a2, w, min_norm=0.0); a2.texts[-1].set_text(""); a2.set_title(f"step {step:,}: {label}", color=INK, fontsize=10.5, pad=4)
    ax.axvline(step, color=MUTED, lw=1, ls=(0, (2, 3)))
savefig(fig, "14b_pentagon_loss.png")
save_json("14_phase_changes.json", results)
print(json.dumps(results, indent=1)[:3800])
