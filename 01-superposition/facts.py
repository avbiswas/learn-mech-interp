"""The person -> sport fact table and the feed-forward block that memorises it, importable.

    x      = arrow(first name) + arrow(last name)      16 numbers, fixed random, never trained
    h      = ReLU(W_in x + b)                          m hidden neurons
    out    = x + W_out h                               the block's output is added back onto its input
    scores = U out                                     one score per sport, the highest wins

This is the shape of a feed-forward layer in a transformer, residual connection included. In the script
the "second weight matrix" W2 is U @ W_out (8 x m): column n is neuron n's vote for each sport, and
scores = W2 h + U x. Removing the residual connection was tried on 2026-10-12 and reverted: without it
the sparse block groups same-sport people on one neuron and no single-person neurons appear.
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


def train(m, relu=True, seed=0, steps=8000, quiet=0.0, log=None, log_every=1000):
    """quiet > 0 adds a penalty on how much the neurons fire, which makes each neuron fire for fewer people."""
    gs = torch.Generator().manual_seed(1000 + seed)           # must differ from the seed that made the name arrows
    P = dict(Win=torch.randn(m, D, generator=gs) * 0.3, b=torch.zeros(m),
             Wout=torch.randn(D, m, generator=gs) * 0.3, U=torch.randn(8, D, generator=gs) * 0.3)
    for p in P.values():
        p.requires_grad_()
    opt = torch.optim.Adam(P.values(), lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        loss = torch.nn.functional.cross_entropy(scores(P, X, relu), y)
        if quiet:
            loss = loss + quiet * hidden(P, X, relu).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if log is not None and (_ % log_every == 0 or _ == steps - 1):
            log(_, loss.item(), {k: v.detach() for k, v in P.items()})
    P = {k: v.detach() for k, v in P.items()}
    return P, scores(P, X, relu).argmax(1)




def train_sparse(m, quiet, seed=0, steps=20000, log=None, log_every=2000):
    """A wide block with a penalty on firing. The penalty is each neuron's activation times the size of
    what it writes, so the block cannot dodge it by shrinking activations and growing W_out."""
    gs = torch.Generator().manual_seed(1000 + seed)
    P = dict(Win=torch.randn(m, D, generator=gs) * 0.3, b=torch.zeros(m),
             Wout=torch.randn(D, m, generator=gs) * (0.3 * (64 / m) ** 0.5), U=torch.randn(8, D, generator=gs) * 0.3)
    for p in P.values():
        p.requires_grad_()
    opt = torch.optim.Adam(P.values(), lr=5e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        h = hidden(P, X)
        loss = torch.nn.functional.cross_entropy((X + h @ P["Wout"].T) @ P["U"].T, y)
        loss = loss + quiet * (h * (P["U"] @ P["Wout"]).norm(dim=0)).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if log is not None and (_ % log_every == 0 or _ == steps - 1):
            log(_, loss.item(), {k: v.detach() for k, v in P.items()})
    P = {k: v.detach() for k, v in P.items()}
    return P, scores(P, X).argmax(1)


def firing(P, x=None):
    """Which neurons fire for which people. A neuron counts as firing when it is above 10% of its own
    strongest firing over the 400 people; neurons that never get above 0.05 are treated as unused."""
    H = hidden(P, X)
    top = H.max(0).values
    Hx = H if x is None else hidden(P, x)
    return (Hx > 0.1 * top.clamp(min=1e-6)) & (top > 0.05)
