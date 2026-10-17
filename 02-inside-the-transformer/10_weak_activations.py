"""Do weak neuron activations matter in a language model? (the "already clean" debate, Oct 2026)

Cammarata (Goodfire) found that in an image model, removing the weakest 75% of activations barely changes accuracy
(72% -> 71%). Nanda and Gao expect the opposite in LLMs: weak activations carry superposed information, especially
memorised facts. Both said LLM evidence was missing. This is a small version of that test on Qwen3-0.6B.

Neuron = one of the 3,072 inputs to an MLP's down_proj (the SwiGLU output; it can be negative). For each neuron we
measure the distribution of |activation| on calibration text, then on held-out text zero every activation below that
neuron's p-th percentile, in all 28 layers at once.

Compared with: zeroing the same fraction of activations at random, and zeroing the strongest 5%.
Measured on held-out Wikipedia paragraphs (HotpotQA contexts): loss, KL to the original next-token distribution,
top-1 agreement with the original model, and accuracy on 40 one-token facts.
"""
import torch, numpy as np, pyarrow.parquet as pq, matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from common import load, blocks, save, r3, SERIES, MUTED, INK
from data.facts import FACTS

tok, model = load("Qwen/Qwen3-0.6B")
NL = len(blocks(model))
HP = hf_hub_download("hotpotqa/hotpot_qa", "distractor/validation-00000-of-00001.parquet", repo_type="dataset",
                     revision="1908d6afbbead072334abe2965f91bd2709910ab")
paras = []
for row in pq.read_table(HP, columns=["context"]).to_pylist()[:200]:
    for sents in row["context"]["sentences"]:
        p = "".join(sents).strip()
        if len(p) > 400:
            paras.append(p)
rng = np.random.RandomState(0); rng.shuffle(paras)
calib, evals = paras[:60], paras[60:140]
enc = lambda t: tok(t, return_tensors="pt", truncation=True, max_length=128).input_ids

# ---------- 1. per-neuron |activation| percentiles on calibration text
store = {L: [] for L in range(NL)}
hooks = [b.mlp.down_proj.register_forward_pre_hook(lambda m, a, L=L: store[L].append(a[0][0, 1:].abs())) for L, b in enumerate(blocks(model))]
for t in calib: model(enc(t))
for h in hooks: h.remove()
PCTS = [50, 75, 90, 95, 99]
thr = {L: torch.quantile(torch.cat(store[L]), torch.tensor([p / 100 for p in PCTS]), dim=0) for L in range(NL)}   # (len(PCTS), 3072)
print("calibration tokens:", sum(x.shape[0] for x in store[0]))
del store

# ---------- 2. ablations
mass = {}


def make_hook(L, mode, k, only=None):
    def hook(m, args):
        a = args[0]; aa = a.abs()
        if only is not None and L != only:
            return None
        if mode == "weak":
            mask = aa < thr[L][k]
        elif mode == "random":
            g = torch.Generator().manual_seed(1000 * L + a.shape[1])
            mask = torch.rand(aa.shape, generator=g) < PCTS[k] / 100
        elif mode == "strong":
            mask = aa >= thr[L][PCTS.index(95)]
        mass.setdefault("removed", 0.0); mass.setdefault("total", 0.0)
        mass["removed"] += (aa * mask)[0, 1:].sum().item(); mass["total"] += aa[0, 1:].sum().item()
        return (a.masked_fill(mask, 0.0),)
    return hook


def evaluate(mode=None, k=None, ref=None, only=None):
    mass.clear()
    hs = [b.mlp.down_proj.register_forward_pre_hook(make_hook(L, mode, k, only)) for L, b in enumerate(blocks(model))] if mode else []
    loss, kl, agree, n, logps = 0, 0, 0, 0, []
    for i, t in enumerate(evals):
        ids = enc(t)
        lp = model(ids).logits[0, :-1].log_softmax(-1)
        y = ids[0, 1:]
        loss += -lp.gather(1, y[:, None]).sum().item(); n += len(y)
        if ref is not None:
            kl += (ref[i].exp() * (ref[i] - lp)).sum().item()
            agree += (ref[i].argmax(-1) == lp.argmax(-1)).sum().item()
        else:
            logps.append(lp)
    facts = sum(int(model(enc(p)).logits[0, -1].argmax() == tok(a).input_ids[0]) for p, a in FACTS) / len(FACTS)
    for h in hs: h.remove()
    out = {"loss": loss / n, "kl": kl / n if ref is not None else 0.0, "top1_agree": agree / n if ref is not None else 1.0,
           "facts": facts, "mass_removed": mass["removed"] / mass["total"] if mass else 0.0}
    return out, logps


base, ref = evaluate()
print(f"original: loss {base['loss']:.3f}, facts {base['facts']:.2f}")
res = {"original": {k: r3(v) for k, v in base.items()}, "weak": {}, "random": {}}
for k, p in enumerate(PCTS):
    for mode in ["weak", "random"]:
        o, _ = evaluate(mode, k, ref)
        res[mode][p] = {kk: r3(v) for kk, v in o.items()}
        print(f"{mode:6s} {p:2d}% of activations zeroed: loss {o['loss']:.3f}  KL {o['kl']:.3f}  top-1 agree {o['top1_agree']:.2f}  facts {o['facts']:.2f}  (|activation| mass removed {o['mass_removed']:.2f})")
o, _ = evaluate("strong", None, ref)
res["strong_5pct"] = {kk: r3(v) for kk, v in o.items()}
print(f"strongest 5% zeroed:          loss {o['loss']:.3f}  KL {o['kl']:.3f}  top-1 agree {o['top1_agree']:.2f}  facts {o['facts']:.2f}  (mass removed {o['mass_removed']:.2f})")

# ---------- 3. one layer at a time: weakest 75% zeroed in a single layer
k75 = PCTS.index(75); res["one_layer_weak75"] = {}
for L in range(0, NL, 3):
    o, _ = evaluate("weak", k75, ref, only=L)
    res["one_layer_weak75"][L] = {kk: r3(v) for kk, v in o.items()}
    print(f"layer {L:2d} only, weakest 75% zeroed: loss {o['loss']:.3f}  KL {o['kl']:.3f}  top-1 agree {o['top1_agree']:.2f}  facts {o['facts']:.2f}")

# ---------- 4. chart
fig, ax = plt.subplots(1, 4, figsize=(21, 4.4))
for c, mode in enumerate(["weak", "random"]):
    lab = "weakest activations zeroed" if mode == "weak" else "same number zeroed at random"
    ax[0].plot(PCTS, [res[mode][p]["loss"] for p in PCTS], "o-", color=SERIES[c], lw=2, label=lab)
    ax[1].plot(PCTS, [res[mode][p]["top1_agree"] for p in PCTS], "o-", color=SERIES[c], lw=2, label=lab)
    ax[2].plot(PCTS, [res[mode][p]["facts"] for p in PCTS], "o-", color=SERIES[c], lw=2, label=lab)
for a, key, title in [(ax[0], "loss", "Loss on Wikipedia text"), (ax[1], "top1_agree", "Same top prediction as the original"),
                      (ax[2], "facts", "One-token facts still correct")]:
    a.axhline(base[key] if key != "top1_agree" else 1.0, color=MUTED, ls="--", lw=1)
    a.scatter([95], [res["strong_5pct"][key]], color=SERIES[3], zorder=5, label="strongest 5% zeroed")
    a.set(xlabel="% of each neuron's activations zeroed", title=title)
    a.legend(frameon=False, fontsize=8)
ax[0].set_yscale("log")
Ls = list(res["one_layer_weak75"])
ax[3].plot(Ls, [res["one_layer_weak75"][L]["facts"] for L in Ls], "o-", color=SERIES[0], lw=2, label="facts correct")
ax[3].plot(Ls, [res["one_layer_weak75"][L]["top1_agree"] for L in Ls], "o-", color=SERIES[2], lw=2, label="same top prediction")
ax[3].axhline(base["facts"], color=MUTED, ls="--", lw=1)
ax[3].set(xlabel="the one layer whose weakest 75% are zeroed", title="One layer at a time", ylim=(0, 1.02)); ax[3].legend(frameon=False, fontsize=8)
save("10_weak_activations", res, fig)
