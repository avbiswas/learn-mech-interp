"""Experiment 9. Where does a fact live?

Question for the viewer: we give a small network 200 facts of the form "person -> sport" to memorize,
but only 64 neurons. Does one neuron end up holding "Michael Jordan plays basketball"?

The network is one transformer feed-forward block, on its own:

    x                         the name, as a vector in a 16-number "stream"   (fixed, random, not trained)
    h   = ReLU(W_in x + b)    64 neurons        (16 -> 64, the 4x expansion real transformers use)
    out = x + W_out h         back to 16 numbers, added onto the stream       (the residual connection)
    scores = U out            one score per sport                             (the read-out)

What is NOT here: attention. In a real LLM, attention builds the name vector from the tokens before
this step, and picks which attribute to read out after it. We hand the block the name vector directly.
"""
import torch
from common import *

D, M, K = 16, 64, 8
SPORTS = ["basketball", "soccer", "tennis", "golf", "cricket", "baseball", "boxing", "swimming"]
REAL = {
    "basketball": ["Michael Jordan", "LeBron James", "Kobe Bryant", "Stephen Curry", "Shaquille O'Neal"],
    "soccer": ["Lionel Messi", "Cristiano Ronaldo", "Pele", "Diego Maradona", "Zinedine Zidane"],
    "tennis": ["Serena Williams", "Roger Federer", "Rafael Nadal", "Novak Djokovic", "Steffi Graf"],
    "golf": ["Tiger Woods", "Jack Nicklaus", "Arnold Palmer", "Rory McIlroy", "Phil Mickelson"],
    "cricket": ["Sachin Tendulkar", "Virat Kohli", "MS Dhoni", "Brian Lara", "Shane Warne"],
    "baseball": ["Babe Ruth", "Derek Jeter", "Hank Aaron", "Jackie Robinson", "Shohei Ohtani"],
    "boxing": ["Muhammad Ali", "Mike Tyson", "Floyd Mayweather", "Manny Pacquiao", "Joe Frazier"],
    "swimming": ["Michael Phelps", "Katie Ledecky", "Mark Spitz", "Ian Thorpe", "Ryan Lochte"],
}
N = 200
g = torch.Generator().manual_seed(0)
names = [n for s in SPORTS for n in REAL[s]]
labels = [SPORTS.index(s) for s in SPORTS for _ in REAL[s]]
for i in range(len(names), N):                       # made-up people fill the list up to 200 facts
    names.append(f"Person {i + 1}"); labels.append(int(torch.randint(0, K, (1,), generator=g)))
y = torch.tensor(labels)
X = torch.nn.functional.normalize(torch.randn(N, D, generator=g), dim=1)    # one fixed vector per name


def train(m, relu=True, seed=0, steps=6000):
    gs = torch.Generator().manual_seed(1000 + seed)       # must differ from the seed that made the name vectors
    P = dict(Win=torch.randn(m, D, generator=gs) * 0.3, b=torch.zeros(m),
             Wout=torch.randn(D, m, generator=gs) * 0.3, U=torch.randn(K, D, generator=gs) * 0.3)
    for p in P.values():
        p.requires_grad_()
    opt = torch.optim.Adam(P.values(), lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        loss = torch.nn.functional.cross_entropy(scores(P, X, relu), y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    P = {k: v.detach() for k, v in P.items()}
    return P, (scores(P, X, relu).argmax(1) == y).float().mean().item()


def hidden(P, x, relu=True):
    h = x @ P["Win"].T + P["b"]
    return torch.relu(h) if relu else h


def scores(P, x, relu=True, keep=None):
    h = hidden(P, x, relu)
    if keep is not None:
        h = h * keep
    return (x + h @ P["Wout"].T) @ P["U"].T


def best(m, relu, seeds=3):
    return max((train(m, relu, s) for s in range(seeds)), key=lambda r: r[1])


# ---- 1. How many facts can it hold, with and without the ReLU? ----
widths = [2, 4, 8, 16, 32, 64, 128]
cap = {"relu": [], "linear": []}
for m in widths:
    for tag in cap:
        cap[tag].append(round(best(m, tag == "relu")[1] * N))
    print(f"{m:4d} neurons: with ReLU {cap['relu'][-1]:3d}/{N} facts   without ReLU {cap['linear'][-1]:3d}/{N}")

# ---- 2. Open up the 64-neuron model ----
P, acc = best(M, True)
H = hidden(P, X)                                             # (200 names, 64 neurons)
fires = H > 1e-6
mj = names.index("Michael Jordan")
order = H[mj].argsort(descending=True)
top = order[0].item()
others = [(names[i], SPORTS[labels[i]], round(H[i, top].item(), 2)) for i in H[:, top].argsort(descending=True).tolist() if i != mj and fires[i, top]]

# Each neuron's push on every sport, for Michael Jordan: activation x (its output column, read by the read-out)
push = H[mj].unsqueeze(1) * (P["Wout"].T @ P["U"].T)         # (64 neurons, 8 sports)
start = X[mj] @ P["U"].T                                     # what the stream says before the block adds anything
firing = [j for j in order.tolist() if fires[mj, j]]          # his firing neurons, strongest first
running = torch.cat([start.unsqueeze(0), start + push[firing].cumsum(0)])       # (firing + 1, 8)
share = (push[:, 0] / push[:, 0].sum()).tolist()               # each neuron's share of the basketball score

# How many of a person's strongest neurons must we switch off before the answer goes wrong?
def knockouts(i, limit=M):
    keep = torch.ones(M)
    for k, j in enumerate(H[i].argsort(descending=True).tolist()[:limit], 1):
        keep[j] = 0
        if scores(P, X[i:i + 1], keep=keep).argmax(1).item() != labels[i]:
            return k
    return limit
ko = torch.tensor([knockouts(i) for i in range(N)])

# The simplest test of "one neuron per fact": switch off any single neuron. How many facts break?
single = [(scores(P, X, keep=torch.ones(M).index_fill(0, torch.tensor([j]), 0)).argmax(1) != y).sum().item() for j in range(M)]

results = dict(
    sizes=dict(stream=D, neurons=M, facts=N, sports=K), facts_recalled=round(acc * N), capacity=dict(neurons=widths, **cap),
    michael_jordan=dict(
        neurons_firing=int(fires[mj].sum()), strongest_neuron=top, strongest_neuron_activation=round(H[mj, top].item(), 2),
        strongest_neuron_also_fires_for=len(others), examples_it_also_fires_for=others[:8],
        score_before_block={s: round(v, 2) for s, v in zip(SPORTS, start.tolist())},
        score_after_block={s: round(v, 2) for s, v in zip(SPORTS, running[-1].tolist())},
        biggest_single_neuron_share_of_basketball_score=round(max(share), 2),
        neurons_pushing_basketball_up=int((push[:, 0] > 0.05).sum()), neurons_pushing_basketball_down=int((push[:, 0] < -0.05).sum()),
        neurons_to_switch_off_before_wrong=int(ko[mj])),
    all_facts=dict(avg_neurons_firing_per_person=round(fires.float().sum(1).mean().item(), 1),
                   avg_people_per_neuron=round(fires.float().sum(0).mean().item(), 1),
                   median_neurons_to_switch_off_before_wrong=int(ko.median()),
                   facts_broken_by_switching_off_one_neuron=dict(worst=max(single), average=round(sum(single) / M, 2))),
    running_scores=[[round(v, 3) for v in r] for r in running.tolist()])
save_json("09_where_does_a_fact_live.json", results)

# ---- Figure A: capacity ----
fig, ax = plt.subplots(figsize=(8.2, 4.2))
ax.plot(range(len(widths)), cap["relu"], color=SERIES[0], lw=2, marker="o", ms=5)
ax.plot(range(len(widths)), cap["linear"], color=SERIES[1], lw=2, marker="o", ms=5)
ax.set_xticks(range(len(widths))); ax.set_xticklabels(widths); ax.set_ylim(0, N * 1.08)
ax.axhline(N, color=MUTED, lw=1, ls=(0, (2, 3))); ax.text(0, N + 4, f"all {N} facts", color=MUTED, fontsize=9.5)
ax.text(len(widths) - 0.9, cap["relu"][-1] - 16, "with ReLU", color=INK, ha="right")
ax.text(len(widths) - 0.9, cap["linear"][-1] + 8, "without ReLU", color=INK, ha="right")
ax.grid(axis="y", lw=0.6); ax.set_axisbelow(True)
ax.set_xlabel("neurons in the block"); ax.set_ylabel("facts recalled correctly")
ax.set_title("How many person → sport facts can one feed-forward block memorize?", color=INK, loc="left", pad=10)
savefig(fig, "09a_facts_vs_neurons.png")

# ---- Figure B: who fires which neuron (the 40 real names) ----
fig, ax = plt.subplots(figsize=(12, 7.2))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ_BLUE)
ax.imshow(H[:40].numpy(), cmap=cmap, vmin=0, vmax=H[:40].max().item(), aspect="auto")
ax.set_yticks(range(40)); ax.set_yticklabels([f"{names[i]}  ·  {SPORTS[labels[i]]}" for i in range(40)], color=INK2, fontsize=8)
ax.set_xticks(range(0, M, 8)); ax.set_xticklabels([f"N{j + 1}" for j in range(0, M, 8)]); ax.tick_params(length=0)
for k in range(1, 8):
    ax.axhline(k * 5 - 0.5, color=SURFACE, lw=3)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xlabel("the 64 neurons")
ax.set_title("Each row is a person, each column a neuron. Bright = that neuron fires for that person.", color=INK, loc="left", pad=10)
savefig(fig, "09b_who_fires_which_neuron.png")

# ---- Figure C: the answer builds up, neuron by neuron ----
fig, ax = plt.subplots(figsize=(8.6, 4.4))
F = len(firing)
for k in range(1, K):
    ax.plot(range(F + 1), running[:, k].numpy(), color=MUTED, lw=1.2)
ax.plot(range(F + 1), running[:, 0].numpy(), color=SERIES[0], lw=2.4)
ax.text(F + 0.5, running[-1, 0].item(), "basketball", color=INK, va="center")
ax.text(F + 0.5, running[-1, 1:].mean().item(), "the other\n7 sports", color=INK2, va="center")
ax.set_xlim(0, F * 1.2); ax.grid(axis="y", lw=0.6); ax.set_axisbelow(True)
ax.set_xlabel("his firing neurons, added in one at a time (strongest first)"); ax.set_ylabel("score for each sport")
ax.set_title("Michael Jordan: the answer is built by adding up many neurons", color=INK, loc="left", pad=10)
savefig(fig, "09c_answer_builds_up.png")

print(f"\n{M}-neuron model recalls {round(acc * N)}/{N} facts")
for k, v in results["michael_jordan"].items():
    print("  MJ", k, ":", v)
for k, v in results["all_facts"].items():
    print("  all", k, ":", v)
lead = (running[:, 0] > running[:, 1:].max(1).values)
print("  MJ: basketball in the lead after adding k neurons:", "".join("Y" if v else "." for v in lead.tolist()))
c = torch.nn.functional.normalize(P["Win"], dim=1) @ X.T
print("  check: max |cos| between a neuron's input weights and any name vector:", round(c.abs().max().item(), 2))

# ---- 3. The 8-neuron model: 200 facts cannot be one-per-neuron when there are only 8 neurons ----
P8, acc8 = best(8, True)
H8 = hidden(P8, X); f8 = H8 > 1e-6
small = dict(facts_recalled=round(acc8 * N), neurons=8,
             people_each_neuron_fires_for=sorted(f8.sum(0).tolist()),
             michael_jordan_fires=[j + 1 for j in range(8) if f8[mj, j]],
             distinct_on_off_patterns_across_people=len({tuple(r) for r in f8.tolist()}))
results["eight_neuron_model"] = small
results["all_facts"]["fewest_and_most_people_per_neuron"] = [int(fires.sum(0).min()), int(fires.sum(0).max())]
save_json("09_where_does_a_fact_live.json", results)
print("  8-neuron model:", small)
print("  64-neuron model, fewest/most people per neuron:", results["all_facts"]["fewest_and_most_people_per_neuron"])
