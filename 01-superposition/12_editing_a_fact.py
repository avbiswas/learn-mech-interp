"""Experiment 12. Can we rewrite one fact by hand?

Question for the viewer: the block has memorised which sport 400 people play. If the facts really live in
its weights, we should be able to change one of them by editing the weights, without retraining and
without disturbing the other 399.

Each neuron is read as a key and a value:
    key   = row of W_in      -> which name pairs make it fire
    value = column of U @ W_out  -> its vote for each of the 8 sports when it fires

The edit is one rank-one change to W_out (the method ROME uses on real language models):
    W_out <- W_out + delta (C^-1 k)^T / (k^T C^-1 k)
k = the neuron pattern for the person being edited, C = how neuron patterns overlap across everyone,
delta = the change that swaps the old sport's score with the new one, mapped back through the read-out U.
The naive version leaves C out and uses k directly.
"""
import json, torch
from common import *
from facts import *

COL = SERIES + ["#8e6fd1", "#5bb5c9", "#a3a05a"]            # one color per sport
ALL_F = torch.arange(len(firsts)).repeat_interleave(len(lasts)); ALL_L = torch.arange(len(lasts)).repeat(len(firsts))
X_ALL = EF[ALL_F] + EL[ALL_L]                                 # every first x last combination, 496 of them
cell = {(int(fi[j]), int(li[j])): j for j in range(N)}


def edit(P, i, new, careful=True):
    H = hidden(P, X); k = H[i]
    C = H.T @ H / len(H) + 1e-3 * torch.eye(H.shape[1])
    d = torch.linalg.solve(C, k) if careful else k
    lo = scores(P, X[i:i + 1])[0]; old = int(lo.argmax())
    want = lo.clone(); want[new], want[old] = lo[old], lo[new]
    delta = torch.linalg.pinv(P["U"]) @ (want - lo)
    Q = dict(P); Q["Wout"] = P["Wout"] + torch.outer(delta, d) / (d @ k)
    return Q


def table(ax, pred, title, mark_wrong=True):
    ax.set_xlim(-0.5, len(lasts) - 0.5); ax.set_ylim(len(firsts) - 0.5, -0.5)
    for (f, l), j in cell.items():
        right = pred[j] == y[j]
        ax.add_patch(plt.Rectangle((l - 0.44, f - 0.44), 0.88, 0.88, color=COL[int(pred[j])], alpha=1.0 if (right or not mark_wrong) else 0.25))
        if j < N_REAL:
            ax.add_patch(plt.Rectangle((l - 0.44, f - 0.44), 0.88, 0.88, fill=False, edgecolor=INK, lw=1.2))
    ax.set_yticks(range(len(firsts))); ax.set_yticklabels(firsts, fontsize=7.5, color=INK2)
    ax.set_xticks(range(len(lasts))); ax.set_xticklabels(lasts, fontsize=7, color=INK2, rotation=90)
    ax.tick_params(length=0); ax.set_title(title, color=INK, pad=8, fontsize=11)
    for s in ax.spines.values():
        s.set_visible(False)


results = {}
P, pred = max((train(64, True, s) for s in range(3)), key=lambda r: (r[1] == y).sum().item())
Plin, pred_lin = train(64, False, 0)
results["recalled"] = dict(relu=int((pred == y).sum()), linear=int((pred_lin == y).sum()), facts=N)

# Figure a: the fact table, and what each block recalls.
fig, axes = plt.subplots(3, 1, figsize=(10.5, 13.5))
table(axes[0], y, f"The facts: {N} people, colored by sport. White outline = a real athlete.", mark_wrong=False)
table(axes[1], pred_lin, f"Block without ReLU: {int((pred_lin == y).sum())} of {N} right (faded = wrong)")
table(axes[2], pred, f"Block with ReLU: {int((pred == y).sum())} of {N} right")
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COL[s]) for s in range(8)], labels=SPORTS, loc="lower center", ncol=8, frameon=False, fontsize=9)
fig.subplots_adjust(hspace=0.42, bottom=0.09)
savefig(fig, "12a_the_fact_table.png")

# Figure b: three neurons as stencils over every first x last combination.
mj = names.index("Michael Jordan")
H, H_ALL = hidden(P, X), hidden(P, X_ALL)
contrib = H[mj] * (P["U"] @ P["Wout"])[int(y[mj])]            # how much each neuron adds to his basketball score
top = contrib.argsort(descending=True)[:3].tolist()
fig, axes = plt.subplots(3, 1, figsize=(10.5, 12.5))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ_BLUE)
results["stencils"] = []
for ax, n in zip(axes, top):
    grid = H_ALL[:, n].view(len(firsts), len(lasts))
    votes = P["U"] @ P["Wout"][:, n]
    ax.imshow(grid.numpy(), cmap=cmap, vmin=0, vmax=max(2.5, grid.max().item()), aspect="auto")
    ax.add_patch(plt.Rectangle((int(li[mj]) - 0.5, int(fi[mj]) - 0.5), 1, 1, fill=False, edgecolor=INK, lw=1.8))
    ax.set_yticks(range(len(firsts))); ax.set_yticklabels(firsts, fontsize=7.5, color=INK2)
    ax.set_xticks(range(len(lasts))); ax.set_xticklabels(lasts, fontsize=7, color=INK2, rotation=90); ax.tick_params(length=0)
    order = votes.argsort(descending=True).tolist()
    ax.set_title(f"Neuron {n}: fires for {int((grid > 0).sum())} of 496 name pairs.  Votes for {SPORTS[order[0]]} ({votes[order[0]]:+.1f}), "
                 f"against {SPORTS[order[-1]]} ({votes[order[-1]]:+.1f})", color=INK, pad=8, fontsize=10.5)
    results["stencils"].append(dict(neuron=n, fires_for_pairs=int((grid > 0).sum()), fires_for_michael_jordan=round(H[mj, n].item(), 2),
                                    share_of_the_push_toward_basketball=round((contrib[n] / contrib.clamp(min=0).sum()).item(), 2),
                                    votes={SPORTS[s]: round(votes[s].item(), 1) for s in range(8)}))
fig.subplots_adjust(hspace=0.5)
savefig(fig, "12b_neurons_as_stencils.png")
results["michael_jordan"] = dict(neurons_firing=int((H[mj] > 0).sum()), of=64)

# The edit: Michael Jordan now plays baseball.
new = SPORTS.index("baseball")
others = torch.arange(N) != mj
results["edit_michael_jordan"] = {}
for label, careful in [("careful", True), ("naive", False)]:
    Q = edit(P, mj, new, careful); p2 = scores(Q, X).argmax(1)
    changed = [dict(name=names[j], was=SPORTS[int(pred[j])], now=SPORTS[int(p2[j])]) for j in range(N) if j != mj and p2[j] != pred[j]]
    results["edit_michael_jordan"][label] = dict(
        now_says=SPORTS[int(p2[mj])], other_facts_changed=len(changed), examples_changed=changed[:6],
        scores_before={SPORTS[s]: round(v, 1) for s, v in enumerate(scores(P, X[mj:mj + 1])[0].tolist())},
        scores_after={SPORTS[s]: round(v, 1) for s, v in enumerate(scores(Q, X[mj:mj + 1])[0].tolist())},
        still_right={n_: SPORTS[int(p2[names.index(n_)])] for n_ in ["Michael Phelps", "Michael Chang", "Magic Johnson", "Dustin Johnson"]},
        largest_weight_change=round((Q["Wout"] - P["Wout"]).abs().max().item(), 3), typical_weight=round(P["Wout"].abs().mean().item(), 3))

# How clean is an edit in general? Edit every one of the 400 people in turn, at several block sizes.
results["edit_everyone"] = {}
sizes, rows = [32, 64, 128, 256], {"careful": [], "naive": []}
for m in sizes:
    stats = {"careful": [], "naive": []}; took = {"careful": 0, "naive": 0}; total = 0
    for seed in range(3):
        Pm, pm = (P, pred) if (m == 64 and seed == 0) else train(m, True, seed)
        if int((pm == y).sum()) < N:
            continue
        for i in range(N):
            tgt = (int(y[i]) + 3) % 8; total += 1
            for label, careful in [("careful", True), ("naive", False)]:
                p2 = scores(edit(Pm, i, tgt, careful), X).argmax(1); o = torch.arange(N) != i
                took[label] += int(p2[i] == tgt); stats[label].append(int((p2[o] != pm[o]).sum()))
    results["edit_everyone"][str(m)] = {}
    for label in stats:
        t = torch.tensor(stats[label], dtype=torch.float32)
        results["edit_everyone"][str(m)][label] = dict(edits=total, edit_took=took[label], other_facts_changed_mean=round(t.mean().item(), 2),
                                                        other_facts_changed_worst=int(t.max()), edits_with_nothing_else_changed=int((t == 0).sum()))
        rows[label].append(t.mean().item())

fig, ax = plt.subplots(figsize=(7.4, 4.3))
for label, c in [("naive", SERIES[1]), ("careful", SERIES[0])]:
    ax.plot(range(len(sizes)), rows[label], color=c, lw=2, marker="o", ms=8, label=f"{label} edit")
    for xpos, v in enumerate(rows[label]):
        ax.text(xpos, v + 4, f"{v:.1f}", ha="center", color=INK, fontsize=9.5)
ax.set_xticks(range(len(sizes))); ax.set_xticklabels([f"{m} neurons" for m in sizes]); ax.set_ylim(-4, max(rows["naive"]) * 1.2)
ax.set_ylabel("other facts changed by one edit (of 399)"); ax.grid(axis="y", lw=0.6); ax.legend(frameon=False)
ax.set_title("Changing one person's sport by hand: how many other facts move?", color=INK, pad=10)
savefig(fig, "12c_editing_a_fact.png")
save_json("12_editing_a_fact.json", results)
for k, v in results.items():
    print(k, json.dumps(v)[:1400])
