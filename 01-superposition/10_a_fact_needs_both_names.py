"""Experiment 10. A fact needs both names.

Question for the viewer: "Michael" could be Jordan or Phelps. "Johnson" could be Magic, Randy, Dustin
or Jack. Neither name on its own tells you the sport. So what does a network need in order to look up
a fact that belongs to the PAIR?

Same feed-forward block as experiment 9. The one change is the input: every first name has an arrow,
every last name has an arrow, and the vector for a person is the two added together.

    x      = arrow(first name) + arrow(last name)      16 numbers, fixed, not trained
    h      = ReLU(W_in x + b)                          64 neurons
    out    = x + W_out h                               added back onto the stream
    scores = U out                                     one score per sport

In a real LLM, attention is what brings the two names together into one vector. Here we add them by hand.
"""
import torch
from common import *

D, M = 16, 64
SPORTS = ["basketball", "soccer", "tennis", "golf", "cricket", "baseball", "boxing", "swimming"]
# Only athletes who share a first or last name with another athlete on the list, so that names are
# reused heavily: 16 first names and 31 last names. (An earlier version used 67 athletes with 47 first
# and 61 last names. Names were reused so little there that "each name votes separately" got 200 of 200.)
REAL = """Michael Jordan,basketball; Magic Johnson,basketball; David Robinson,basketball; Chris Paul,basketball
Kevin Durant,basketball; Anthony Davis,basketball
David Beckham,soccer; Ryan Giggs,soccer
Roger Federer,tennis; Chris Evert,tennis; Michael Chang,tennis
Jack Nicklaus,golf; Dustin Johnson,golf; Ian Poulter,golf
Shane Warne,cricket; Chris Gayle,cricket; Kevin Pietersen,cricket; Ian Botham,cricket; Mark Waugh,cricket; Joe Root,cricket
Jackie Robinson,baseball; Randy Johnson,baseball; David Ortiz,baseball; Mark McGwire,baseball
Joe DiMaggio,baseball; Roger Clemens,baseball
Joe Frazier,boxing; Jack Johnson,boxing; Jack Dempsey,boxing; Shane Mosley,boxing; Anthony Joshua,boxing
Michael Phelps,swimming; Mark Spitz,swimming; Ian Thorpe,swimming; Ryan Lochte,swimming"""
people = []
for item in REAL.replace("\n", ";").split(";"):
    name, sport = item.strip().split(",")
    first, last = name.split(" ", 1)
    people.append((first, last, SPORTS.index(sport)))
N_REAL, N = len(people), 400
firsts = sorted({p[0] for p in people}); lasts = sorted({p[1] for p in people})
g = torch.Generator().manual_seed(0)
taken = {(p[0], p[1]) for p in people}
while len(people) < N:                                # made-up people: a real first name with a real last name, random sport
    f = firsts[int(torch.randint(0, len(firsts), (1,), generator=g))]
    l = lasts[int(torch.randint(0, len(lasts), (1,), generator=g))]
    if (f, l) not in taken:
        taken.add((f, l)); people.append((f, l, int(torch.randint(0, 8, (1,), generator=g))))
names = [f"{f} {l}" for f, l, _ in people]
fi = torch.tensor([firsts.index(p[0]) for p in people]); li = torch.tensor([lasts.index(p[1]) for p in people])
y = torch.tensor([p[2] for p in people])
EF = torch.nn.functional.normalize(torch.randn(len(firsts), D, generator=g), dim=1)     # first-name arrows
EL = torch.nn.functional.normalize(torch.randn(len(lasts), D, generator=g), dim=1)      # last-name arrows
X = EF[fi] + EL[li]


def hidden(P, x, relu=True):
    h = x @ P["Win"].T + P["b"]
    return torch.relu(h) if relu else h


def scores(P, x, relu=True, keep=None):
    h = hidden(P, x, relu)
    if keep is not None:
        h = h * keep
    return (x + h @ P["Wout"].T) @ P["U"].T


def train(m, relu=True, seed=0, steps=8000):
    gs = torch.Generator().manual_seed(1000 + seed)           # must differ from the seed that made the name arrows
    P = dict(Win=torch.randn(m, D, generator=gs) * 0.3, b=torch.zeros(m),
             Wout=torch.randn(D, m, generator=gs) * 0.3, U=torch.randn(8, D, generator=gs) * 0.3)
    for p in P.values():
        p.requires_grad_()
    opt = torch.optim.Adam(P.values(), lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        loss = torch.nn.functional.cross_entropy(scores(P, X, relu), y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    P = {k: v.detach() for k, v in P.items()}
    return P, scores(P, X, relu).argmax(1)


def best(m, relu, seeds=3):
    return max((train(m, relu, s) for s in range(seeds)), key=lambda r: (r[1] == y).sum().item())


def train_votes(steps=4000):
    """The best any 'each name votes separately' method could do, with unlimited room:
    a free table of sport scores per first name, plus a free table per last name. No block, no bottleneck."""
    A = torch.zeros(len(firsts), 8, requires_grad=True); B = torch.zeros(len(lasts), 8, requires_grad=True)
    opt = torch.optim.Adam([A, B], lr=5e-2)
    for _ in range(steps):
        loss = torch.nn.functional.cross_entropy(A[fi] + B[li], y)
        opt.zero_grad(); loss.backward(); opt.step()
    return (A[fi] + B[li]).argmax(1).detach()


shares = torch.tensor([sum(1 for q in people if (q[0] == p[0] or q[1] == p[1]) and q[2] != p[2]) for p in people])
votes = train_votes()
widths = [4, 8, 16, 32, 64, 128]
cap = {"relu": [], "linear": []}
for m in widths:
    for tag in cap:
        cap[tag].append(int((best(m, tag == "relu")[1] == y).sum()))
    print(f"{m:4d} neurons: with ReLU {cap['relu'][-1]:3d}/{N}   without ReLU {cap['linear'][-1]:3d}/{N}")
print(f"each name votes separately, unlimited room: {int((votes == y).sum())}/{N}")

P, pred = best(M, True)
Plin, pred_lin = best(M, False)
H = hidden(P, X); fires = H > 1e-6


def show(name):
    i = names.index(name)
    return dict(truth=SPORTS[y[i]], with_relu=SPORTS[pred[i]], without_relu=SPORTS[pred_lin[i]], names_voting_separately=SPORTS[votes[i]])


# Is a person's most useful neuron a "both names together" detector?
# Compare how hard it fires for the person, for people sharing only one of the names, and for everyone else.
push_all = H.unsqueeze(2) * (P["Wout"].T @ P["U"].T).unsqueeze(0)                 # (people, neurons, sports)
own = push_all[torch.arange(N), :, y]                                              # each neuron's push on the right sport
topn = own.argmax(1)
rows = []
for i in range(N):
    j = topn[i].item()
    one = [k for k in range(N) if k != i and (people[k][0] == people[i][0]) != (people[k][1] == people[i][1])]
    none = [k for k in range(N) if people[k][0] != people[i][0] and people[k][1] != people[i][1]]
    rows.append((H[i, j].item(), H[one, j].mean().item() if one else float("nan"), H[none, j].mean().item()))
rows = torch.tensor(rows)
share_top = (own.max(1).values / own.clamp(min=0).sum(1))


def knockouts(i):
    keep = torch.ones(M)
    for k, j in enumerate(H[i].argsort(descending=True).tolist(), 1):
        keep[j] = 0
        if scores(P, X[i:i + 1], keep=keep).argmax(1).item() != y[i].item():
            return k
    return M
ko = torch.tensor([knockouts(i) for i in range(N)])
single = [(scores(P, X, keep=torch.ones(M).index_fill(0, torch.tensor([j]), 0)).argmax(1) != y).sum().item() for j in range(M)]
mj = names.index("Michael Jordan")
wrong_lin = pred_lin != y
results = dict(
    sizes=dict(stream=D, neurons=M, facts=N, real_people=N_REAL, first_names=len(firsts), last_names=len(lasts)),
    facts_recalled=dict(with_relu=int((pred == y).sum()), without_relu=int((pred_lin == y).sum()),
                        names_voting_separately_unlimited_room=int((votes == y).sum())),
    capacity=dict(neurons=widths, **cap),
    examples={n: show(n) for n in names[:N_REAL]},
    who_the_linear_block_gets_wrong=dict(
        avg_conflicting_name_sharers_when_wrong=round(shares[wrong_lin].float().mean().item(), 2),
        avg_conflicting_name_sharers_when_right=round(shares[~wrong_lin].float().mean().item(), 2)),
    neurons=dict(avg_neurons_firing_per_person=round(fires.float().sum(1).mean().item(), 1),
                 avg_people_per_neuron=round(fires.float().sum(0).mean().item(), 1),
                 fewest_and_most_people_per_neuron=[int(fires.sum(0).min()), int(fires.sum(0).max())],
                 top_neuron_activation_for_the_person=round(rows[:, 0].mean().item(), 2),
                 same_neuron_for_people_sharing_one_name=round(rows[:, 1].nanmean().item(), 2),
                 same_neuron_for_people_sharing_no_name=round(rows[:, 2].mean().item(), 2),
                 median_share_of_score_from_top_neuron=round(share_top.median().item(), 2),
                 median_neurons_to_switch_off_before_wrong=int(ko.median()),
                 facts_broken_by_switching_off_one_neuron=dict(worst=max(single), average=round(sum(single) / M, 2))),
    michael_jordan=dict(neurons_firing=int(fires[mj].sum()), share_of_score_from_top_neuron=round(share_top[mj].item(), 2),
                        neurons_to_switch_off_before_wrong=int(ko[mj]),
                        top_neuron_fires_for=dict(Michael_Jordan=round(H[mj, topn[mj]].item(), 2),
                                                  **{n.replace(" ", "_"): round(H[names.index(n), topn[mj]].item(), 2)
                                                     for n in names if n != "Michael Jordan" and (n.startswith("Michael ") or n.endswith(" Jordan"))})))
save_json("10_a_fact_needs_both_names.json", results)

# ---- Figure: three ways to look up the fact ----
fig, ax = plt.subplots(figsize=(8.4, 3.6))
vals = [results["facts_recalled"]["without_relu"], results["facts_recalled"]["names_voting_separately_unlimited_room"], results["facts_recalled"]["with_relu"]]
labs = ["Block without ReLU", "Each name votes separately\n(unlimited room)", "Block with ReLU"]
ax.barh(range(3), vals, height=0.5, color=[SERIES[1], SERIES[1], SERIES[0]])
for k, v in enumerate(vals):
    ax.text(v + 3, k, f"{v} of {N}", va="center", color=INK)
ax.set_yticks(range(3)); ax.set_yticklabels(labs, color=INK2); ax.invert_yaxis(); ax.set_xlim(0, N * 1.18)
ax.axvline(N, color=MUTED, lw=1, ls=(0, (2, 3))); ax.grid(axis="x", lw=0.6); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
ax.set_xlabel("facts recalled correctly")
ax.set_title("A fact belongs to the pair of names, not to either name", color=INK, loc="left", pad=10)
savefig(fig, "10_a_fact_needs_both_names.png")

import json
print(json.dumps({k: results[k] for k in ["sizes", "facts_recalled", "examples", "who_the_linear_block_gets_wrong", "neurons", "michael_jordan"]}, indent=1))

# ---- Reading the block from its weights alone (no inputs run through it) ----
# A neuron's input is W_in . (first arrow + last arrow) + b, so it splits exactly into
#   key_first[first name] + key_last[last name] + b      and the neuron fires when that sum is above 0.
# Its output column, read through U, is a fixed set of votes for the sports.
KF = EF @ P["Win"].T                       # (first names, neurons): what each neuron reads from each first name
KL = EL @ P["Win"].T                       # (last names, neurons)
V = (P["Wout"].T @ P["U"].T)               # (neurons, sports): each neuron's votes
def card(j):
    f = KF[:, j].argsort(descending=True).tolist(); l = KL[:, j].argsort(descending=True).tolist()
    return dict(neuron=j + 1, bias=round(P["b"][j].item(), 2),
                first_names_it_listens_for=[(firsts[k], round(KF[k, j].item(), 2)) for k in f[:4]],
                first_names_that_shut_it_off=[(firsts[k], round(KF[k, j].item(), 2)) for k in f[-3:]],
                last_names_it_listens_for=[(lasts[k], round(KL[k, j].item(), 2)) for k in l[:4]],
                last_names_that_shut_it_off=[(lasts[k], round(KL[k, j].item(), 2)) for k in l[-3:]],
                votes={s: round(v, 1) for s, v in zip(SPORTS, V[j].tolist())})
mj_neurons = own[mj].argsort(descending=True)[:3].tolist()
top_vote_share = (V.abs().max(1).values / V.abs().sum(1))
weights_view = dict(
    michael_jordan_top3_neurons=[card(j) for j in mj_neurons],
    michael_jordan_readout=[dict(neuron=j + 1, from_Michael=round(KF[firsts.index("Michael"), j].item(), 2),
                                 from_Jordan=round(KL[lasts.index("Jordan"), j].item(), 2), bias=round(P["b"][j].item(), 2),
                                 total=round((KF[firsts.index("Michael"), j] + KL[lasts.index("Jordan"), j] + P["b"][j]).item(), 2),
                                 basketball_vote=round(V[j, 0].item(), 1)) for j in mj_neurons],
    how_one_sided_are_votes=dict(median_share_of_a_neurons_votes_going_to_its_top_sport=round(top_vote_share.median().item(), 2),
                                 if_votes_were_spread_evenly=round(1 / 8, 2)),
    neurons_needing_both_names=int(((KF.max(0).values + P["b"] < 0) & (KL.max(0).values + P["b"] < 0)).sum()),
    neurons_one_name_alone_can_fire=int(((KF.max(0).values + P["b"] > 0) | (KL.max(0).values + P["b"] > 0)).sum()))
results["weights_view"] = weights_view
save_json("10_a_fact_needs_both_names.json", results)
print(json.dumps(weights_view, indent=1))
