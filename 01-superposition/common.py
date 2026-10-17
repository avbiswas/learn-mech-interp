"""Shared pieces for the superposition experiments.

The toy model (from "Toy Models of Superposition"):

    x   = how much of each concept is in the input      (n numbers)
    h   = W x                                           (m numbers, m < n)
    x'  = ReLU(W^T h + b)                               (n numbers, should match x)

Column i of W is the direction where concept i lives. train_toy() trains many independent
copies at once (one per entry of `sparsity`), so sweeps and restarts are a single call.
"""
import json, math, os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

# Dark chart theme. Concept colors are fixed: a concept keeps its color in every figure.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#1a1a19", "#f0efec", "#c3c2b7", "#898781", "#2c2c2a", "#383835"
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]
SEQ_BLUE = ["#1a1a19", "#104281", "#256abf", "#5598e7", "#b7d3f6"]   # surface -> light, for magnitude
CONCEPTS = ["cat", "dog", "car", "tree", "boat"]                      # most important first

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "grid.color": GRID, "font.size": 11, "axes.titlesize": 12,
    "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans",
})


def sample(n, sparsity, batch, g):
    """Each concept is absent with probability `sparsity`, otherwise present with strength in [0, 1].
    sparsity has shape (M,) -> returns (M, batch, n)."""
    M = sparsity.shape[0]
    present = torch.rand(M, batch, n, generator=g) > sparsity.view(M, 1, 1)
    return torch.rand(M, batch, n, generator=g) * present


def forward(W, b, x, relu_out=True, hidden_relu=False):
    h = x @ W.transpose(1, 2)                 # (M, batch, m)
    if hidden_relu:
        h = torch.relu(h)
    out = h @ W + b.unsqueeze(1)              # (M, batch, n)
    return (torch.relu(out) if relu_out else out), h


def train_toy(n, m, sparsity, importance, relu_out=True, hidden_relu=False,
              steps=6000, batch=1024, lr=5e-3, seed=0, bias=True, log=None, log_every=500):
    """Train M = len(sparsity) independent toy models in parallel. Returns W (M, m, n), b (M, n), loss (M,).
    `log(step, loss, W)` is called every `log_every` steps when given; it only reads, so results are unchanged."""
    g = torch.Generator().manual_seed(seed)
    sparsity = torch.as_tensor(sparsity, dtype=torch.float32)
    importance = torch.as_tensor(importance, dtype=torch.float32)
    M = sparsity.shape[0]
    W = (torch.randn(M, m, n, generator=g) * 0.3).requires_grad_()
    b = torch.zeros(M, n, requires_grad=True)
    opt = torch.optim.Adam([W, b] if bias else [W], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for step in range(steps):
        x = sample(n, sparsity, batch, g)
        out, _ = forward(W, b, x, relu_out, hidden_relu)
        loss = (importance * (x - out) ** 2).sum(-1).mean(-1)     # (M,)
        opt.zero_grad(); loss.sum().backward(); opt.step(); sched.step()
        if log is not None and (step % log_every == 0 or step == steps - 1):
            log(step, loss.detach(), W.detach())
    with torch.no_grad():
        x = sample(n, sparsity, 20000, g)
        out, _ = forward(W, b, x, relu_out, hidden_relu)
        final = (importance * (x - out) ** 2).sum(-1).mean(-1)
    return W.detach(), b.detach(), final


def best_of(restarts, sparsity, **kw):
    """Run `restarts` seeds for every sparsity and keep the lowest-loss model for each (as the paper did)."""
    S = torch.as_tensor(sparsity, dtype=torch.float32)
    W, b, loss = train_toy(sparsity=S.repeat(restarts), **kw)
    k = S.shape[0]
    idx = loss.view(restarts, k).argmin(0) * k + torch.arange(k)
    return W[idx], b[idx], loss[idx]


def save_json(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1)


def savefig(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join("out", name))


def draw_arrows(ax, W, names=CONCEPTS, lim=1.5, min_norm=0.15):
    """W is (2, n). One arrow per concept; concepts the model dropped are listed underneath."""
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axhline(0, color=GRID, lw=1); ax.axvline(0, color=GRID, lw=1)
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, ls=(0, (3, 4)), color=AXIS, lw=1))
    # The hidden layer here is linear, so the whole picture can be rotated freely without changing the
    # model. We turn every plot so the first concept points up, which makes the panels comparable.
    t = math.pi / 2 - math.atan2(W[1, 0].item(), W[0, 0].item())
    R = torch.tensor([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    W = R @ W
    side = next((i for i in range(1, W.shape[1]) if abs(W[0, i].item()) > 0.3), 1)
    if W[0, side] < 0:                   # mirror so the first concept that is off the vertical sits on the right
        W = W * torch.tensor([[-1.0], [1.0]])
    dropped = []
    for i, name in enumerate(names):
        x, y = W[0, i].item(), W[1, i].item()
        r = (x * x + y * y) ** 0.5
        if r < min_norm:
            dropped.append(name); continue
        ax.annotate("", xy=(x, y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=SERIES[i], lw=2.2, mutation_scale=14))
        ax.text(x * (1 + 0.24 / r), y * (1 + 0.24 / r), name, color=INK, ha="center", va="center", fontsize=11)
    ax.text(0, -lim * 0.97, ("dropped: " + ", ".join(dropped)) if dropped else "nothing dropped",
            color=MUTED, ha="center", va="bottom", fontsize=9.5)
    return dropped
