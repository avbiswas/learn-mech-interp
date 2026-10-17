"""D2. Where does Qwen3-0.6B look up "Michael Jordan -> basketball"? Activation patching (causal tracing).

Clean run:     "Michael Jordan plays the sport of"  -> basketball
Corrupted run: "Michael Phelps plays the sport of"  -> swimming
Copy ONE activation (one layer, one token position) from the clean run into the corrupted run and measure how much of
the basketball-vs-swimming logit difference comes back (0 = still Phelps, 1 = fully Jordan).
Done for the residual stream, the attention output and the MLP output of every layer at every position.
"""
import torch, matplotlib.pyplot as plt
from common import load, blocks, save, r3

tok, model = load("Qwen/Qwen3-0.6B")
clean, corrupt = "Michael Jordan plays the sport of", "Michael Phelps plays the sport of"
A, B = tok(" basketball").input_ids[0], tok(" swimming").input_ids[0]
ci, xi = tok(clean, return_tensors="pt").input_ids, tok(corrupt, return_tensors="pt").input_ids
toks = [tok.decode(t) for t in ci[0]]
NL, T = len(blocks(model)), ci.shape[1]
diff = lambda lg: (lg[0, -1, A] - lg[0, -1, B]).item()

SITES = {"residual stream": lambda b: b, "attention output": lambda b: b.self_attn, "MLP output": lambda b: b.mlp}
first = lambda o: o[0] if isinstance(o, tuple) else o


def run(ids, site=None, layer=None, pos=None, cache=None, record=None):
    hs = []
    for L, b in enumerate(blocks(model)):
        for name, get in SITES.items():
            mod = get(b)
            if record is not None:
                hs.append(mod.register_forward_hook(lambda m, i, o, k=(name, L): record.__setitem__(k, first(o).clone())))
            if site == name and L == layer:
                def patch(m, i, o, k=(name, L)):
                    x = first(o).clone(); x[:, pos] = cache[k][:, pos]
                    return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x
                hs.append(mod.register_forward_hook(patch))
    lg = model(ids).logits
    for h in hs: h.remove()
    return lg


cache = {}
d_clean, d_corrupt = diff(run(ci, record=cache)), diff(run(xi))
print(f"logit diff basketball - swimming: clean {d_clean:.2f}, corrupted {d_corrupt:.2f}")
res = {"tokens": toks, "clean_diff": r3(d_clean), "corrupt_diff": r3(d_corrupt)}
fig, ax = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
for k, name in enumerate(SITES):
    M = torch.zeros(NL, T)
    for L in range(NL):
        for p in range(T):
            M[L, p] = (diff(run(xi, name, L, p, cache)) - d_corrupt) / (d_clean - d_corrupt)
    res[name] = [[r3(v) for v in r] for r in M.tolist()]
    best = divmod(int(M.argmax()), T)
    print(f"{name:17s} strongest: layer {best[0]}, token {toks[best[1]]!r}, recovers {M.max():.2f}")
    im = ax[k].imshow(M, aspect="auto", cmap="magma", vmin=0, vmax=1, origin="lower")
    ax[k].set_xticks(range(T), toks, rotation=45, ha="right"); ax[k].set(title=f"patch {name}", xlabel="token")
ax[0].set_ylabel("layer")
plt.colorbar(im, ax=ax, fraction=0.02, label="share of 'Jordan' answer restored")
save("05_michael_jordan_patching", res, fig)
