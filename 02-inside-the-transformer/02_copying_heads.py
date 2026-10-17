"""B1b. Read each GPT-2 head's job from its weights alone: does its OV circuit copy tokens?

OV circuit through the vocabulary: W_E · W_V · W_O · W_U (a vocab x vocab table, rank 64).
  - eigenvalue copying score: sum(Re λ) / sum(|λ|), from the 64x64 matrix W_V (W_E^T... ) trick (AB and BA share eigenvalues)
  - a plain check: for 2,000 common tokens, is the token's OWN logit in the top 10 outputs of the table?
    This check uses the "effective embedding" (W_E + MLP0, as in McDougall et al. 2023), the head's ln_1, and centres
    each output column (removes what the head writes no matter what it reads). Without these, a few odd tokens dominate.
The eigenvalue score ignores layer norm (as in the framework paper). Induction scores come from repeated random tokens, for comparison.
"""
import torch, matplotlib.pyplot as plt
from common import load, save, r3, SERIES, MUTED, INK

tok, model = load("gpt2", eager=True)
NL, NH, DH, D = 12, 12, 64, 768
W_E = model.transformer.wte.weight            # (V, D); GPT-2 ties W_U = W_E^T
EE = W_E.T @ W_E                               # (D, D)
sample = torch.arange(1000, 3000)              # 2,000 ordinary tokens
mlp0 = lambda x: x + model.transformer.h[0].mlp(model.transformer.h[0].ln_2(x))
X = mlp0(W_E[sample])                          # effective embeddings

eig, top1, examples = torch.zeros(NL, NH), torch.zeros(NL, NH), {}
for l, b in enumerate(model.transformer.h):
    Wv = b.attn.c_attn.weight[:, 2 * D:]       # (D, D) -> per head (D, 64)
    Wo = b.attn.c_proj.weight                  # (D, D) -> per head (64, D)
    for h in range(NH):
        V, O = Wv[:, h * DH:(h + 1) * DH], Wo[h * DH:(h + 1) * DH]
        ev = torch.linalg.eigvals((O @ EE @ V).double())          # 64x64, same non-zero eigenvalues as the vocab table
        eig[l, h] = ev.real.sum() / ev.abs().sum()
        table = b.ln_1(X) @ V @ O @ W_E.T                          # (2000, V): row = token attended to, cols = logit change
        table = table - table.mean(0, keepdim=True)
        own = table[torch.arange(len(sample)), sample]
        top1[l, h] = ((table > own[:, None]).sum(-1) < 10).float().mean()

# induction score, for overlaying
torch.manual_seed(0)
half = torch.randint(1000, 20000, (8, 50)); ids = torch.cat([half, half], 1)
att = model(ids, output_attentions=True).attentions
ind = torch.stack([torch.stack([a[:, :, q, q - 49] for q in range(50, 100)]).mean((0, 1)) for a in att])

# example: what does the strongest copying induction head write when it attends to a few words?
for l, h in [(6, 9), (9, 9), (4, 7)]:
    b = model.transformer.h[l]
    V, O = b.attn.c_attn.weight[:, 2 * D + h * DH: 2 * D + (h + 1) * DH], b.attn.c_proj.weight[h * DH:(h + 1) * DH]
    base = (b.ln_1(X) @ V @ O @ W_E.T).mean(0)
    for w in [" Paris", " Jordan", " banana", " seven"]:
        t = tok(w).input_ids[0]
        out = b.ln_1(mlp0(W_E[t])) @ V @ O @ W_E.T - base
        examples[f"{l}.{h}{w}"] = [tok.decode(i) for i in out.topk(5).indices]

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.1]})
im = ax[0].imshow(eig, cmap="berlin", vmin=-1, vmax=1)
ax[0].set(xlabel="head", ylabel="layer", title="Copying score Σλ / Σ|λ| (red = copies, blue = suppresses)")
plt.colorbar(im, ax=ax[0], fraction=0.046)
x, y = eig.flatten(), ind.flatten()
ax[1].scatter(x, y, s=18, color=SERIES[0], alpha=0.7)
for i in y.topk(6).indices:
    ax[1].annotate(f"{i // NH}.{i % NH}", (x[i], y[i]), xytext=(4, 2), textcoords="offset points", fontsize=9, color=INK)
ax[1].axvline(0, color=MUTED, lw=1, ls="--")
ax[1].set(xlabel="copying score (from weights only)", ylabel="induction score (from running the model)", title="Induction heads are copying heads")
save("02_copying_heads", {
    "eig_score": [[r3(v) for v in r] for r in eig.tolist()], "own_token_in_top10": [[r3(v) for v in r] for r in top1.tolist()],
    "induction_score": [[r3(v) for v in r] for r in ind.tolist()], "heads_over_0.5": int((eig > 0.5).sum()), "heads_under_-0.5": int((eig < -0.5).sum()),
    "what_heads_write": examples}, fig)
print("heads >0.5:", int((eig > 0.5).sum()), " < -0.5:", int((eig < -0.5).sum()))
for i in ind.flatten().topk(6).indices:
    print(f"induction head {i // NH}.{i % NH}: copy {eig.flatten()[i]:.2f}, own-token-in-top10 {top1.flatten()[i]:.2f}")
print("most negative:", [(f"{int(i) // NH}.{int(i) % NH}", r3(eig.flatten()[i])) for i in (-eig.flatten()).topk(5).indices])
for k, v in examples.items(): print(k, "->", v)
