"""Train tiny attention-only transformers from scratch and watch induction heads form.

B4. Ported from book/scripts/ch4_phase_change.py (the website's version). Writes out/04_induction_phase_change.{json,png}.

Data: uniform random tokens (so bigram statistics are useless), with spans copied from earlier in the
same sequence at random offsets. The only way to beat chance on a copied token is to look back,
find the earlier copy of the current token, and read what came next: induction.
"""
import json, math, torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)
V, N, DM, NH = 64, 96, 64, 4
DH = DM // NH

def batch(B, g):
    x = torch.randint(0, V, (B, N), generator=g)
    copied = torch.zeros(B, N, dtype=torch.bool)
    for b in range(B):
        for _ in range(6):
            L = int(torch.randint(6, 16, (1,), generator=g))
            dst = int(torch.randint(L + 4, N - L, (1,), generator=g))
            src = int(torch.randint(0, dst - L, (1,), generator=g))
            x[b, dst:dst + L] = x[b, src:src + L]
            copied[b, dst + 1:dst + L] = True       # the first token of a copy is not predictable
    return x, copied

class Attn(nn.Module):
    def __init__(s):
        super().__init__(); s.ln = nn.LayerNorm(DM); s.qkv = nn.Linear(DM, 3 * DM); s.o = nn.Linear(DM, DM)
    def forward(s, x):
        B, T, _ = x.shape
        q, k, v = s.qkv(s.ln(x)).view(B, T, 3, NH, DH).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(DH)
        att = att.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf")).softmax(-1)
        s.pattern = att.detach()
        return x + s.o((att @ v).transpose(1, 2).reshape(B, T, DM))

class Model(nn.Module):
    def __init__(s, layers):
        super().__init__()
        s.emb = nn.Embedding(V, DM); s.pos = nn.Embedding(N, DM)
        s.layers = nn.ModuleList([Attn() for _ in range(layers)]); s.ln = nn.LayerNorm(DM); s.un = nn.Linear(DM, V)
    def forward(s, x):
        h = s.emb(x) + s.pos(torch.arange(x.shape[1]))
        for l in s.layers: h = l(h)
        return s.un(s.ln(h))

def probe(m):
    """Same scores as for GPT-2: a random sequence of 40 tokens repeated twice (padded to N)."""
    g = torch.Generator().manual_seed(123)
    T = 40
    r = torch.randint(0, V, (64, T), generator=g)
    x = torch.cat([r, r, torch.randint(0, V, (64, N - 2 * T), generator=g)], 1)
    with torch.no_grad():
        logits = m(x)
    q = torch.arange(T + 1, 2 * T)
    lp = F.log_softmax(logits, -1)
    rep_loss = -lp[:, q - 1].gather(-1, x[:, q, None]).mean().item()     # loss on predictable repeated tokens
    prev, ind = [], []
    for l in m.layers:
        P = l.pattern
        prev.append(P[:, :, torch.arange(1, 2 * T), torch.arange(0, 2 * T - 1)].mean((0, 2)).tolist())
        ind.append(P[:, :, q, q - T + 1].mean((0, 2)).tolist())
    return rep_loss, prev, ind

def train(layers, steps=6000):
    torch.manual_seed(0)
    m = Model(layers); opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=0.01)
    g = torch.Generator().manual_seed(7)
    log = []
    for step in range(steps + 1):
        x, copied = batch(64, g)
        logits = m(x)[:, :-1]
        tl = F.cross_entropy(logits.reshape(-1, V), x[:, 1:].reshape(-1), reduction="none").view(64, N - 1)
        loss = tl.mean()
        if step % 25 == 0:
            c = copied[:, 1:]
            rl, prev, ind = probe(m)
            log.append({"step": step, "loss": round(loss.item(), 4),
                        "loss_copied": round(tl[c].mean().item(), 4), "loss_fresh": round(tl[~c].mean().item(), 4),
                        "probe_rep_loss": round(rl, 4),
                        "prev": [[round(v, 3) for v in r] for r in prev], "ind": [[round(v, 3) for v in r] for r in ind]})
            if step % 500 == 0: print(layers, step, log[-1]["loss"], log[-1]["loss_copied"], [max(r) for r in log[-1]["ind"]])
        opt.zero_grad(); loss.backward(); opt.step()
    rl, prev, ind = probe(m)
    with torch.no_grad():
        g2 = torch.Generator().manual_seed(123); T = 40
        r = torch.randint(0, V, (1, 12), generator=g2)
        x = torch.cat([r, r], 1); m(F.pad(x, (0, 0)))
        pats = [[[round(v, 3) for v in row] for row in l.pattern[0, h].tolist()] for l in m.layers for h in range(NH)]
    return {"log": log, "example_tokens": x[0].tolist(), "example_patterns": pats}

from common import save, SERIES, MUTED
import matplotlib.pyplot as plt
torch.set_grad_enabled(True)
two, one = train(2), train(1)
out = {"V": V, "N": N, "d_model": DM, "heads": NH, "chance_loss": round(math.log(V), 4), "two_layer": two, "one_layer": one}
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for c, (lab, run) in enumerate([("2 layers", two), ("1 layer", one)]):
    st = [r["step"] for r in run["log"]]
    ax[0].plot(st, [r["loss_copied"] for r in run["log"]], color=SERIES[c], lw=2, label=f"{lab}: loss on copied tokens")
    ax[1].plot(st, [max(max(h) for h in r["ind"]) for r in run["log"]], color=SERIES[c], lw=2, label=f"{lab}: best induction score")
ax[0].axhline(math.log(V), color=MUTED, ls="--", lw=1); ax[0].text(50, math.log(V) + 0.05, "chance", color=MUTED, fontsize=9)
ax[0].set(xlabel="training step", ylabel="loss", title="Loss drops suddenly, but only with 2 layers")
ax[1].set(xlabel="training step", ylabel="attention to the right earlier token", title="…at the moment an induction head forms", ylim=(0, 1))
for a in ax: a.legend(frameon=False, fontsize=9)
save("04_induction_phase_change", out, fig)
