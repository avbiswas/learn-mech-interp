"""A1. At which layer does the model 'know' the answer? Logit lens over 40 one-token facts, GPT-2 vs Qwen3-0.6B.

For each fact the model gets right: probability and rank of the correct answer after every layer
(final norm + unembedding applied to the residual stream at the last position).
"""
import torch, matplotlib.pyplot as plt
from common import load, final_norm, save, r3, SERIES, MUTED
from data.facts import FACTS

res, fig, ax = {}, *plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={"width_ratios": [1, 1, 0.9]})
for c, name in enumerate(["gpt2", "Qwen/Qwen3-0.6B"]):
    tok, model = load(name)
    probs, ranks, kept = [], [], []
    for prompt, ans in FACTS:
        t = tok(ans).input_ids[0]
        out = model(tok(prompt, return_tensors="pt").input_ids, output_hidden_states=True)
        if out.logits[0, -1].argmax() != t:
            continue
        hs = out.hidden_states
        lg = [model.lm_head(final_norm(model)(h[0, -1])) for h in hs[:-1]] + [out.logits[0, -1]]
        p = torch.stack([l.softmax(-1)[t] for l in lg])
        r = torch.stack([(l > l[t]).sum() for l in lg])
        probs.append(p); ranks.append(r); kept.append(prompt + ans)
    P, R = torch.stack(probs), torch.stack(ranks)
    first_top1 = [int((r == 0).nonzero()[0]) for r in R]
    nl = P.shape[1] - 1
    res[name] = {"n_correct": len(kept), "n_total": len(FACTS), "n_layers": nl, "facts": kept,
                 "prob_by_layer": [[r3(x) for x in p] for p in P.tolist()], "first_layer_top1": first_top1,
                 "median_first_layer_top1_frac": r3(torch.tensor(first_top1).float().median() / nl)}
    print(name, f"{len(kept)}/{len(FACTS)} correct; answer first becomes top-1 at layer (median)",
          int(torch.tensor(first_top1).float().median()), "of", nl)
    if c == 1:
        order = torch.tensor(first_top1).argsort()
        im = ax[0].imshow(P[order], aspect="auto", cmap="magma", vmin=0, vmax=1)
        ax[0].set_yticks(range(len(kept)), [kept[i].split(" ")[-1] for i in order], fontsize=7)
        ax[0].set(xlabel="layer", title=f"{name}: probability of the right answer, per layer")
        plt.colorbar(im, ax=ax[0], fraction=0.04)
        ax[1].imshow(torch.log10(R[order].float() + 1), aspect="auto", cmap="magma_r", vmin=0, vmax=5)
        ax[1].set_yticks([]); ax[1].set(xlabel="layer", title="log10(rank + 1) of the right answer (light = top)")
    depth = torch.arange(nl + 1) / nl
    ax[2].plot(depth, P.mean(0), color=SERIES[c], lw=2, label=f"{name.split('/')[-1]} mean prob")
    ax[2].plot(depth, (R == 0).float().mean(0), color=SERIES[c], lw=2, ls="--", label=f"{name.split('/')[-1]} share at top-1")
ax[2].set(xlabel="depth (0 = embeddings, 1 = output)", title="The answer appears late, then sharpens", ylim=(0, 1.02))
ax[2].legend(frameon=False, fontsize=9)
save("03_logit_lens", res, fig)
