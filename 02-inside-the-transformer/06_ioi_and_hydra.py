"""D3 + D4. Indirect object identification in GPT-2 small, over 100 prompts, and the Hydra effect.

Clean:     "When Mary and John went to the store, John gave a drink to"  -> Mary  (IO = Mary, S = John)
Corrupted: same template with three unrelated names ("When Alice and Bob ..., Kate gave ...")
Metric:    logit(IO) - logit(S) at the last position, averaged over prompts.

1. Direct logit attribution (DLA): what each head's output, at the last position, writes along W_U[IO] - W_U[S].
2. Activation patching: replace one head's output (all positions) with its output on the corrupted prompt.
   How much of the clean logit difference is lost? (total effect of the head)
3. Hydra: compare each head's direct effect with its total effect. If later heads compensate, total < direct.
   For the strongest name mover, show which heads change their DLA when it is knocked out.
"""
import random, torch, matplotlib.pyplot as plt
from common import load, save, r3, SERIES, MUTED, INK

tok, model = load("gpt2", eager=True)
NL, NH, DH = 12, 12, 64
random.seed(0)
NAMES = [n for n in ["Mary", "John", "Alice", "Bob", "Kate", "Tom", "Sarah", "James", "Emma", "David", "Anna", "Paul", "Lisa", "Mark",
                     "Laura", "Peter", "Rachel", "Daniel", "Jessica", "Michael", "Sophie", "Chris", "Helen", "Steve"] if len(tok(" " + n).input_ids) == 1]
TEMPLATES = ["When {A} and {B} went to the store, {S} gave a drink to", "When {A} and {B} went to the park, {S} gave a ball to",
             "While {A} and {B} were at the office, {S} sent an email to", "When {A} and {B} arrived at the party, {S} gave a gift to",
             "When {A} and {B} went to the beach, {S} threw a ball to", "When {A} and {B} were at the house, {S} gave a key to",
             "After {A} and {B} went to the shop, {S} gave a bag to"]          # all 14 tokens long, names at the same positions
clean, corrupt, io, s = [], [], [], []
for i in range(100):
    t = TEMPLATES[i % len(TEMPLATES)]
    a, b, c, d, e = random.sample(NAMES, 5)
    S, IO = (a, b) if i % 2 else (b, a)          # balance which position the subject is in
    clean.append(t.format(A=a, B=b, S=S)); corrupt.append(t.format(A=c, B=d, S=e))
    io.append(tok(" " + IO).input_ids[0]); s.append(tok(" " + S).input_ids[0])
C = tok(clean, return_tensors="pt").input_ids; X = tok(corrupt, return_tensors="pt").input_ids
assert C.shape == X.shape, "templates must tokenize to the same length"
io, s = torch.tensor(io), torch.tensor(s)
direction = model.lm_head.weight[io] - model.lm_head.weight[s]           # (100, D)
metric = lambda lg: (lg[:, -1].gather(1, io[:, None]) - lg[:, -1].gather(1, s[:, None])).mean().item()


def run(ids, record=None, replace=None):
    """record: dict filled with per-head outputs z (B, T, NH, DH) per layer. replace: {(l, h): z tensor} patched in."""
    hs = []
    for l, b in enumerate(model.transformer.h):
        def pre(m, args, l=l):
            z = args[0].view(*args[0].shape[:2], NH, DH)
            if record is not None: record[l] = z.clone()
            if replace:
                z = z.clone()
                for (L, h), v in replace.items():
                    if L == l: z[:, :, h] = v
                return (z.view(*args[0].shape),)
        hs.append(b.attn.c_proj.register_forward_pre_hook(pre))
    lg = model(ids).logits
    for h in hs: h.remove()
    return lg


def dla(z, ln_scale):
    """Direct effect of each head on the metric: head output at the last position, through the final layer-norm scale."""
    out = torch.zeros(NL, NH)
    for l in range(NL):
        Wo = model.transformer.h[l].attn.c_proj.weight.view(NH, DH, -1)
        for h in range(NH):
            v = z[l][:, -1, h] @ Wo[h] / ln_scale                      # (B, D)
            out[l, h] = ((v * model.transformer.ln_f.weight) * direction).sum(-1).mean()
    return out


def final_ln_scale(ids, replace=None):
    store = {}
    hk = model.transformer.ln_f.register_forward_hook(lambda m, i, o: store.__setitem__("x", i[0]))
    run(ids, replace=replace); hk.remove()
    x = store["x"][:, -1]
    return (x - x.mean(-1, keepdim=True)).pow(2).mean(-1, keepdim=True).add(1e-5).sqrt()


zc, zx = {}, {}
m_clean, m_corrupt = metric(run(C, record=zc)), metric(run(X, record=zx))
print(f"logit diff IO - S: clean {m_clean:.2f}, corrupted {m_corrupt:.2f}")
direct = dla(zc, final_ln_scale(C))

total = torch.zeros(NL, NH)
for l in range(NL):
    for h in range(NH):
        total[l, h] = m_clean - metric(run(C, replace={(l, h): zx[l][:, :, h]}))     # how much is LOST when this head is corrupted
key = sorted(((total[l, h].item(), f"{l}.{h}") for l in range(NL) for h in range(NH)), reverse=True)[:12]
print("biggest total effects:", [(n, r3(v)) for v, n in key])
print("biggest direct effects:", [(f"{int(i) // NH}.{int(i) % NH}", r3(direct.flatten()[i])) for i in direct.flatten().abs().topk(10).indices])

# Hydra: knock out the head with the biggest direct effect; who picks up the slack?
top = int(direct.flatten().argmax()); tl, th = divmod(top, NH)
zk = {}
m_knock = metric(run(C, record=zk, replace={(tl, th): zx[tl][:, :, th]}))
direct_after = dla(zk, final_ln_scale(C, {(tl, th): zx[tl][:, :, th]}))
change = direct_after - direct
comp = [(f"{int(i) // NH}.{int(i) % NH}", r3(change.flatten()[i])) for i in change.flatten().abs().topk(8).indices]
print(f"knock out {tl}.{th}: its direct effect {direct[tl, th]:.2f}, actual drop {m_clean - m_knock:.2f}; biggest DLA changes elsewhere: {comp}")

fig, ax = plt.subplots(1, 3, figsize=(19, 5)); fig.subplots_adjust(wspace=0.35)
lim = direct.abs().max()
im = ax[0].imshow(direct, cmap="berlin", vmin=-lim, vmax=lim); ax[0].set(title="Direct effect on 'IO minus S' (red = pushes the right name)", xlabel="head", ylabel="layer"); plt.colorbar(im, ax=ax[0], fraction=0.046)
lim = total.abs().max()
im = ax[1].imshow(total, cmap="berlin", vmin=-lim, vmax=lim); ax[1].set(title="Total effect: logit diff lost when patched", xlabel="head"); plt.colorbar(im, ax=ax[1], fraction=0.046)
x, y = direct.flatten(), total.flatten()
ax[2].scatter(x, y, s=16, color=SERIES[0], alpha=0.7)
lo, hi = min(x.min(), y.min()).item(), max(x.max(), y.max()).item()
ax[2].plot([lo, hi], [lo, hi], color=MUTED, ls="--", lw=1)
for i in (x.abs() + y.abs()).topk(8).indices:
    ax[2].annotate(f"{i // NH}.{i % NH}", (x[i], y[i]), xytext=(4, 2), textcoords="offset points", fontsize=9, color=INK)
ax[2].set(xlabel="direct effect (what the head writes)", ylabel="total effect (what is lost when it is patched)", title="Hydra: points below the line get repaired")
save("06_ioi_and_hydra", {"names": NAMES, "clean_diff": r3(m_clean), "corrupt_diff": r3(m_corrupt),
                          "direct": [[r3(v) for v in r] for r in direct.tolist()], "total": [[r3(v) for v in r] for r in total.tolist()],
                          "knockout": {"head": f"{tl}.{th}", "direct": r3(direct[tl, th]), "actual_drop": r3(m_clean - m_knock),
                                       "dla_change": [[r3(v) for v in r] for r in change.tolist()], "biggest_changes": comp}}, fig)
