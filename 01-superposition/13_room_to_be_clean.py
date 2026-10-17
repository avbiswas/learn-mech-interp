"""Experiment 13. Give the block room and a reason to stay quiet.

Question for the viewer: the 64-neuron block spread Michael Jordan over 31 neurons that each fire for
about half of all the people. What if the same block had 600 neurons, and was also trained to fire as
few of them as it can? (This is the idea behind a sparse autoencoder, applied to the fact block itself.)

Same data and same 16-number input as experiment 12. Two blocks are compared:
    dense : 64 neurons, no penalty
    sparse: 600 neurons, with a penalty on firing
"""
import json, torch
from common import *
from facts import *

M, QUIET, STEPS = 600, 0.15, 20000
print("Experiment 13: where does a fact live?  400 people -> which sport they play")
print(f"dense block : 64 neurons, no penalty")
print(f"sparse block: {M} neurons, penalised for firing (strength {QUIET})\n")


def progress(tag):
    def log(step, loss, P):
        with torch.no_grad():
            right = int((scores(P, X).argmax(1) == y).sum())
            A = firing(P)
            per_person = A.sum(1).float().median().item()
            mjn = int(A[names.index("Michael Jordan")].sum())
        print(f"  {tag} step {step:>6} | loss {loss:7.4f} | recalls {right:>3}/400 | "
              f"neurons per person {per_person:5.1f} | Michael Jordan uses {mjn:>3} neurons", flush=True)
    return log


print("=== training the dense block (64 neurons)")
dense, pred_d = max((train(64, True, s, log=progress(f"seed {s}"), log_every=2000) for s in range(3)),
                    key=lambda r: (r[1] == y).sum().item())
print("\n=== training the sparse block (600 neurons, penalised for firing)")
runs = [train_sparse(M, QUIET, seed, STEPS, log=progress(f"seed {seed}"), log_every=4000) for seed in range(3)]
print()
mj = names.index("Michael Jordan")
allF = torch.arange(len(firsts)).repeat_interleave(len(lasts)); allL = torch.arange(len(lasts)).repeat(len(firsts))
X_ALL = EF[allF] + EL[allL]                                      # all 496 first x last combinations, 96 never seen in training


def describe(P, pred):
    A = firing(P); used = A.any(0); ppn = A.sum(0)
    C = hidden(P, X) * (P["U"] @ P["Wout"])[y]                    # each neuron's push toward each person's correct sport
    top = C.argmax(1)
    solo = sum(int(scores(P, X[i:i + 1], keep=torch.nn.functional.one_hot(top[i], P["b"].numel()).float()).argmax(1) == y[i]) for i in range(N))
    one_sport = sum(1 for n in torch.where(used)[0].tolist() if len(set(y[A[:, n]].tolist())) == 1)
    return dict(recalled=int((pred == y).sum()), neurons=P["b"].numel(), neurons_used=int(used.sum()),
                people_per_neuron_median=float(ppn[used].float().median()), people_per_neuron_max=int(ppn.max()),
                neurons_per_person_median=float(A.sum(1).float().median()),
                neurons_firing_for_exactly_one_person=int((ppn == 1).sum()), neurons_whose_people_all_play_one_sport=one_sport,
                people_whose_most_helpful_neuron_fires_only_for_them=int((ppn[top] == 1).sum()),
                right_answers_from_the_most_helpful_neuron_alone=solo,
                michael_jordan=dict(neurons_firing=int(A[mj].sum()), most_helpful_neuron=int(top[mj]),
                                    that_neuron_fires_for=[names[j] for j in torch.where(A[:, top[mj]])[0].tolist()],
                                    share_of_push_toward_basketball=round((C[mj, top[mj]] / C[mj].clamp(min=0).sum()).item(), 2)))


results = {"settings": dict(sparse_neurons=M, penalty=QUIET, steps=STEPS), "dense_64": describe(dense, pred_d),
           "sparse_by_seed": [describe(P, p) for P, p in runs]}
P, pred = runs[0]
results["sparse"] = results["sparse_by_seed"][0]
n = results["sparse"]["michael_jordan"]["most_helpful_neuron"]
votes = P["U"] @ P["Wout"][:, n]
unseen = firing(P, X_ALL)[:, n].view(len(firsts), len(lasts))
seen = torch.zeros(len(firsts), len(lasts), dtype=torch.bool); seen[fi, li] = True
results["michael_jordan_neuron"] = dict(
    neuron=n, votes={SPORTS[s]: round(votes[s].item(), 1) for s in range(8)},
    fires_for_name_pairs_never_seen_in_training=[f"{firsts[i]} {lasts[j]}" for i, j in (unseen & ~seen).nonzero().tolist()])

# Editing the fact in the sparse block: change only what that one neuron writes.
Q = dict(P); lo = scores(P, X[mj:mj + 1])[0]; want = lo.clone(); b_, bb = SPORTS.index("baseball"), SPORTS.index("basketball")
want[b_], want[bb] = lo[bb], lo[b_]
h = hidden(P, X[mj])[n]
Wn = P["Wout"].clone(); Wn[:, n] += torch.linalg.pinv(P["U"]) @ (want - lo) / h; Q["Wout"] = Wn
p2 = scores(Q, X).argmax(1)
results["edit_one_neuron"] = dict(michael_jordan_now=SPORTS[int(p2[mj])], other_facts_changed=int(((p2 != pred) & (torch.arange(N) != mj)).sum()),
                                  weights_touched=D, weights_in_block=int(P["Wout"].numel() + P["Win"].numel()))

# Figure: Michael Jordan's most helpful neuron in each block, drawn over every name pair, and how shared the neurons are.
fig = plt.figure(figsize=(11, 9.6)); gs_ = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8], hspace=0.62, wspace=0.18)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ_BLUE)
for row, (Pb, title) in enumerate([(dense, "64 neurons"), (P, f"{M} neurons, trained to fire as little as possible")]):
    ax = fig.add_subplot(gs_[row, :])
    C = hidden(Pb, X) * (Pb["U"] @ Pb["Wout"])[y]; nb = int(C[mj].argmax())
    grid = hidden(Pb, X_ALL)[:, nb].view(len(firsts), len(lasts))
    ax.imshow(grid.numpy(), cmap=cmap, vmin=0, vmax=grid.max().item(), aspect="auto")
    ax.add_patch(plt.Rectangle((int(li[mj]) - 0.5, int(fi[mj]) - 0.5), 1, 1, fill=False, edgecolor=INK, lw=1.6))
    ax.set_yticks(range(len(firsts))); ax.set_yticklabels(firsts, fontsize=7, color=INK2)
    ax.set_xticks(range(len(lasts))); ax.set_xticklabels(lasts, fontsize=7, color=INK2, rotation=90); ax.tick_params(length=0)
    k = int(firing(Pb)[:, nb].sum())
    ax.set_title(f"{title}: Michael Jordan's most helpful neuron fires for {k} of the 400 people", color=INK, fontsize=11, pad=8)
for col, (Pb, title, c) in enumerate([(dense, "64 neurons", SERIES[1]), (P, f"{M} neurons, sparse", SERIES[0])]):
    ax = fig.add_subplot(gs_[2, col]); A = firing(Pb); ppn = A.sum(0)[A.any(0)]
    ax.hist(ppn.numpy(), bins=range(0, 320, 10) if col == 0 else [x - 0.5 for x in range(1, 9)], color=c, rwidth=0.9)
    ax.set_xlabel("people one neuron fires for"); ax.set_ylabel("neurons"); ax.grid(axis="y", lw=0.6); ax.set_axisbelow(True)
    ax.set_title(f"{title}: median {int(ppn.float().median())}", color=INK, fontsize=11)
savefig(fig, "13_room_to_be_clean.png")
save_json("13_room_to_be_clean.json", results)
for tag, r in (("DENSE  (64 neurons)", results["dense_64"]), (f"SPARSE ({M} neurons)", results["sparse"])):
    m_ = r["michael_jordan"]
    print(f"=== {tag}")
    print(f"  facts recalled                 : {r['recalled']}/400")
    print(f"  people per neuron (median)     : {r['people_per_neuron_median']:.0f}")
    print(f"  neurons firing for one person  : {r['neurons_firing_for_exactly_one_person']}")
    print(f"  Michael Jordan fires           : {m_['neurons_firing']} neurons")
    print(f"  his most helpful neuron fires for {len(m_['that_neuron_fires_for'])} people: "
          + ", ".join(m_['that_neuron_fires_for'][:8]) + (" ..." if len(m_['that_neuron_fires_for']) > 8 else ""))
    print()
mn = results["michael_jordan_neuron"]
print(f"=== the sparse block's Michael Jordan neuron (#{mn['neuron']}) votes:")
for sport, v in sorted(mn["votes"].items(), key=lambda kv: -kv[1]):
    print(f"  {sport:<12} {v:6.1f}")
ed = results["edit_one_neuron"]
print(f"\n=== editing only that one neuron: Michael Jordan now plays {ed['michael_jordan_now']}, "
      f"other facts changed: {ed['other_facts_changed']}")
