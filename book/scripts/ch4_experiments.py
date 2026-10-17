"""Real experiments on GPT-2 small for Chapter 4 (induction heads, MLPs, logit/tuned lens).

Run with:  python book/scripts/ch4_experiments.py <wikitext-103 validation parquet> <tuned-lens gpt2 params.pt>
Writes book/assets/ch4_data.js (window.CH4 = {...}).

Inputs:
  - wikitext-103-raw-v1 validation split (Salesforce/wikitext on Hugging Face), used as ordinary text.
  - the pretrained tuned lens for gpt2 (AlignmentResearch/tuned-lens, lens/gpt2/params.pt).
"""
import json, sys, random, math
import numpy as np, torch
import pyarrow.parquet as pq
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

torch.set_grad_enabled(False)
torch.manual_seed(0); random.seed(0); np.random.seed(0)
tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()
blocks = model.transformer.h
NL, NH, DH, D = 12, 12, 64, 768
W_E = model.transformer.wte.weight          # (V, D); GPT-2 ties W_U = W_E^T
W_U = model.lm_head.weight.T                # (D, V)
OUT = {}
r3 = lambda x: round(float(x), 3)

# ---------------------------------------------------------------- hooks
def run(ids, ablate=None, attn=False, cache=None, mlp_mean=None):
    """ablate: set of (layer, head) whose output z is zeroed. mlp_mean: {layer: vector} to mean-ablate an MLP.
    cache: dict filled with resid (hidden states), attn_out, mlp_out per layer."""
    hs = []
    if ablate:
        for L in {l for l, _ in ablate}:
            heads = [h for l, h in ablate if l == L]
            def pre(m, i, heads=heads):
                z = i[0].clone()
                for h in heads: z[..., h * DH:(h + 1) * DH] = 0
                return (z,)
            hs.append(blocks[L].attn.c_proj.register_forward_pre_hook(pre))
    if mlp_mean:
        for L, v in mlp_mean.items():
            hs.append(blocks[L].mlp.register_forward_hook(lambda m, i, o, v=v: v.expand_as(o)))
    if cache is not None:
        cache["attn"], cache["mlp"] = {}, {}
        for L, b in enumerate(blocks):
            hs.append(b.attn.register_forward_hook(lambda m, i, o, L=L: cache["attn"].__setitem__(L, o[0])))
            hs.append(b.mlp.register_forward_hook(lambda m, i, o, L=L: cache["mlp"].__setitem__(L, o)))
    try:
        out = model(ids, output_attentions=attn, output_hidden_states=cache is not None)
    finally:
        for h in hs: h.remove()
    if cache is not None: cache["resid"] = out.hidden_states
    return out

def tok_loss(logits, ids):
    lp = torch.log_softmax(logits[:, :-1].float(), -1)
    return -lp.gather(-1, ids[:, 1:, None])[..., 0]          # (B, T-1); entry t = loss predicting token t+1

# ================================================================ 1. repeated random tokens
T, B = 50, 64
g = torch.Generator().manual_seed(1)
rand = torch.randint(1000, 20000, (B, T), generator=g)
bos = torch.full((B, 1), tok.bos_token_id)
rep = torch.cat([bos, rand, rand], 1)                       # positions: 0 = BOS, 1..50 first copy, 51..100 second copy
out = run(rep, attn=True)
loss = tok_loss(out.logits, rep).mean(0)                    # loss[t] = loss for predicting position t+1
OUT["rep_loss"] = [r3(x) for x in loss.tolist()]
OUT["rep_T"] = T
first, second = loss[1:T].mean().item(), loss[T + 1:].mean().item()
OUT["rep_loss_first"], OUT["rep_loss_second"] = r3(first), r3(second)
print("repeated tokens: first-half loss", first, "second-half", second)

A = torch.stack(out.attentions)                              # (L, B, H, Tq, Tk)
q2 = torch.arange(T + 1, 2 * T + 1)                          # queries in the second copy
prev = A[:, :, :, torch.arange(2, 2 * T + 1), torch.arange(1, 2 * T)].mean((1, 3))   # attend to i-1 (skip BOS)
induct = A[:, :, :, q2, q2 - T + 1].mean((1, 3))             # attend to the token AFTER the earlier copy
dup = A[:, :, :, q2, q2 - T].mean((1, 3))                    # attend to the earlier copy itself
OUT["score_prev"] = [[r3(x) for x in row] for row in prev.tolist()]
OUT["score_induction"] = [[r3(x) for x in row] for row in induct.tolist()]
OUT["score_dup"] = [[r3(x) for x in row] for row in dup.tolist()]
ind_heads = [(l, h) for l in range(NL) for h in range(NH) if induct[l, h] > 0.4]
prev_heads = [(l, h) for l in range(NL) for h in range(NH) if prev[l, h] > 0.4]
ind_heads.sort(key=lambda x: -induct[x].item()); prev_heads.sort(key=lambda x: -prev[x].item())
OUT["induction_heads"] = [f"{l}.{h}" for l, h in ind_heads]
OUT["prev_heads"] = [f"{l}.{h}" for l, h in prev_heads]
print("induction heads", OUT["induction_heads"], "prev heads", OUT["prev_heads"])

# ---------------------------------------------------------------- 1b. a readable example (single-token words)
words = [" apple", " river", " green", " table", " music", " cloud", " paper", " horse", " glass", " stone",
         " dream", " storm", " chair", " light", " bread", " ocean"]
ex_words = [" green", " horse", " paper", " storm", " bread", " river", " glass", " music"]
ex = [tok.bos_token_id] + [tok.encode(w)[0] for w in ex_words] * 2
ex_ids = torch.tensor([ex])
o = run(ex_ids, attn=True)
best_ind, best_prev = ind_heads[0], prev_heads[0]
OUT["example_tokens"] = [tok.decode([t]) for t in ex]
OUT["example_attn"] = {f"{l}.{h}": [[r3(x) for x in r] for r in o.attentions[l][0, h].tolist()]
                      for l, h in {best_ind, best_prev, *ind_heads[:3], *prev_heads[:2]}}
probs = torch.softmax(o.logits[0].float(), -1)
OUT["example_pred"] = [[tok.decode([int(i)]), r3(probs[t, i])] for t, i in enumerate(probs.argmax(-1))]

# ---------------------------------------------------------------- 1c. ablations on the repeated-token task
def second_half_loss(ablate=None, ids=rep):
    return tok_loss(run(ids, ablate=ablate).logits, ids)[:, T + 1:].mean().item()
base = second_half_loss()
per_head = torch.zeros(NL, NH)
for l in range(NL):
    for h in range(NH):
        per_head[l, h] = second_half_loss({(l, h)}) - base
OUT["ablate_each_head"] = [[r3(x) for x in r] for r in per_head.tolist()]
rng = random.Random(0)
others = [(l, h) for l in range(NL) for h in range(NH) if (l, h) not in ind_heads and (l, h) not in prev_heads]
rand_sets = [set(rng.sample(others, len(ind_heads))) for _ in range(5)]
OUT["ablate_groups"] = {
    "none": r3(base),
    "induction": r3(second_half_loss(set(ind_heads))),
    "prev": r3(second_half_loss(set(prev_heads))),
    "random_same_size": r3(np.mean([second_half_loss(s) for s in rand_sets])),
    "first_half_baseline": r3(first),
}
print("ablation groups", OUT["ablate_groups"])

# ---------------------------------------------------------------- 1d. K-composition: remove previous-token heads, re-measure induction scores
o2 = run(rep, ablate=set(prev_heads), attn=True)
A2 = torch.stack(o2.attentions)
induct2 = A2[:, :, :, q2, q2 - T + 1].mean((1, 3))
OUT["induction_after_prev_ablation"] = {f"{l}.{h}": [r3(induct[l, h]), r3(induct2[l, h])] for l, h in ind_heads}
print("K-comp check", OUT["induction_after_prev_ablation"])

# ---------------------------------------------------------------- 1e. OV copying score (positive-eigenvalue fraction of W_U W_OV W_E), ignoring layer norm
EE = W_E.T @ W_E                                             # (D, D) = W_U W_E with tied weights, contracted over vocab
copy = torch.zeros(NL, NH)
for l in range(NL):
    Wv = blocks[l].attn.c_attn.weight[:, 2 * D:]
    Wo = blocks[l].attn.c_proj.weight
    for h in range(NH):
        M = Wo[h * DH:(h + 1) * DH] @ EE @ Wv[:, h * DH:(h + 1) * DH]     # (64, 64), same nonzero eigenvalues
        ev = torch.linalg.eigvals(M.double())
        copy[l, h] = (ev.real.sum() / ev.abs().sum()).item()
OUT["score_copy"] = [[r3(x) for x in r] for r in copy.tolist()]
print("copy scores of induction heads", [(f"{l}.{h}", r3(copy[l, h])) for l, h in ind_heads])

# ================================================================ 2. wikitext corpus
texts = [t for t in pq.read_table(sys.argv[1]).column("text").to_pylist() if len(t.strip()) > 200 and not t.strip().startswith("=")]
corpus = tok.encode(" ".join(texts))
CTX = 128
nchunk = min(len(corpus) // CTX, 1600)
chunks = torch.tensor(corpus[:nchunk * CTX]).view(nchunk, CTX)
print("corpus chunks", nchunk)

# ---------------------------------------------------------------- 3. logit lens and tuned lens
lens = torch.load(sys.argv[2], weights_only=True, map_location="cpu")
def read(h, layer, kind):
    if kind == "tuned" and layer < NL:
        h = h + h @ lens[f"{layer}.weight"].T + lens[f"{layer}.bias"]
    return model.lm_head(model.transformer.ln_f(h))

prompts = [
    ("The Eiffel Tower is located in the city of", " Paris"),
    ("The capital of Japan is the city of", " Tokyo"),
    ("Barack Obama was born in the state of", " Hawaii"),
    ("Water is made of hydrogen and", " oxygen"),
    ("The opposite of hot is", " cold"),
]
OUT["lens_prompts"] = []
for p, ans in prompts:
    ids = tok(p, return_tensors="pt").input_ids
    c = {}; run(ids, cache=c)
    a = tok.encode(ans)[0]
    entry = {"prompt": p, "answer": ans, "tokens": [tok.decode([t]) for t in ids[0]], "logit": [], "tuned": []}
    for kind in ("logit", "tuned"):
        for layer in range(NL + 1):        # hidden_states[0] = embeddings, [12] = after final block (already ln_f'd in HF)
            h = c["resid"][layer][0, -1]
            if layer == NL:
                lg = model.lm_head(h)      # HF applies ln_f to the last hidden state
            else:
                lg = read(h, layer, kind)
            pr = torch.softmax(lg.float(), -1)
            top = pr.topk(3)
            rank = int((pr > pr[a]).sum())
            entry[kind].append({"top": [[tok.decode([int(i)]), r3(v)] for v, i in zip(top.values, top.indices)],
                                "p_ans": round(pr[a].item(), 4), "rank": rank})
    OUT["lens_prompts"].append(entry)
    print(p, [e["top"][0][0] for e in entry["logit"]], [e["top"][0][0] for e in entry["tuned"]])

# aggregate: KL(final || lens) and top-1 agreement with the final prediction, per layer, over the corpus
kl = {"logit": np.zeros(NL), "tuned": np.zeros(NL)}; agree = {"logit": np.zeros(NL), "tuned": np.zeros(NL)}
ninput = 0
for i in range(0, 200, 8):
    ids = chunks[i:i + 8]; c = {}; o = run(ids, cache=c)
    fin = torch.log_softmax(o.logits.float(), -1); finp = fin.exp(); top1 = fin.argmax(-1)
    for kind in ("logit", "tuned"):
        for layer in range(NL):
            lp = torch.log_softmax(read(c["resid"][layer], layer, kind).float(), -1)
            kl[kind][layer] += (finp * (fin - lp)).sum(-1).sum().item()
            agree[kind][layer] += (lp.argmax(-1) == top1).sum().item()
    ninput += ids.numel()
OUT["lens_aggregate"] = {k: {"kl": [r3(x / ninput) for x in kl[k]], "agree": [r3(x / ninput) for x in agree[k]]} for k in kl}
print("lens aggregate", OUT["lens_aggregate"])

# ================================================================ 4. MLPs
# 4a. where does the answer logit come from? direct logit attribution per block, with ln_f's scale frozen
fact_prompts = [
    ("The Eiffel Tower is located in the city of", " Paris"), ("The capital of Japan is the city of", " Tokyo"),
    ("The Colosseum is located in the city of", " Rome"), ("The Statue of Liberty is in the city of", " New"),
    ("Big Ben is a clock tower in the city of", " London"), ("The official language of France is", " French"),
    ("The official language of Germany is", " German"), ("Mount Everest is in the country of", " Nepal"),
    ("The Great Wall is in the country of", " China"), ("Michael Jordan played the sport of", " basketball"),
    ("Lionel Messi plays the sport of", " soccer"), ("The Taj Mahal is located in the country of", " India"),
    ("The currency of Japan is the", " yen"), ("The Kremlin is located in the city of", " Moscow"),
    ("Toyota is a car company from", " Japan"), ("The Pyramids of Giza are in the country of", " Egypt"),
]
def dla(ids, ans):
    c = {}; o = run(ids, cache=c)
    ln = model.transformer.ln_f
    x = c["resid"][NL - 1][0, -1]
    # recompute the pre-ln_f final residual: last hidden_state in HF is post-ln_f, so rebuild it
    pre = c["resid"][NL - 1][0, -1] + c["attn"][NL - 1][0, -1] + c["mlp"][NL - 1][0, -1]
    scale = (pre - pre.mean()).pow(2).mean().add(ln.eps).sqrt()
    u = W_U[:, ans] - W_U.mean(1)                                    # logit of the answer, relative to the mean logit
    def contrib(v): return (((v - v.mean()) / scale) * ln.weight) @ u
    emb = c["resid"][0][0, -1]
    return {"embed": contrib(emb).item(),
            "attn": [contrib(c["attn"][l][0, -1]).item() for l in range(NL)],
            "mlp": [contrib(c["mlp"][l][0, -1]).item() for l in range(NL)]}, o
dl_all = []
OUT["dla_prompts"] = []
for p, ans in fact_prompts:
    ids = tok(p, return_tensors="pt").input_ids; a = tok.encode(ans)[0]
    d, o = dla(ids, a)
    top = tok.decode([int(o.logits[0, -1].argmax())])
    dl_all.append(d)
    OUT["dla_prompts"].append({"prompt": p, "answer": ans, "model_top": top,
                               "embed": r3(d["embed"]), "attn": [r3(x) for x in d["attn"]], "mlp": [r3(x) for x in d["mlp"]]})
OUT["dla_mean"] = {"attn": [r3(np.mean([d["attn"][l] for d in dl_all])) for l in range(NL)],
                   "mlp": [r3(np.mean([d["mlp"][l] for d in dl_all])) for l in range(NL)]}
print("DLA mean", OUT["dla_mean"])

# 4b. mean-ablate one MLP layer at a time; how much does the answer's probability fall?
mlp_means = {}
c = {}; run(chunks[:32], cache=c)
for l in range(NL): mlp_means[l] = c["mlp"][l].mean((0, 1))
abl = np.zeros((len(fact_prompts), NL)); base_p = []
for k, (p, ans) in enumerate(fact_prompts):
    ids = tok(p, return_tensors="pt").input_ids; a = tok.encode(ans)[0]
    p0 = torch.log_softmax(run(ids).logits[0, -1].float(), -1)[a].item(); base_p.append(math.exp(p0))
    for l in range(NL):
        p1 = torch.log_softmax(run(ids, mlp_mean={l: mlp_means[l]}).logits[0, -1].float(), -1)[a].item()
        abl[k, l] = p1 - p0
OUT["mlp_ablation_dlogp"] = [r3(x) for x in abl.mean(0)]
OUT["mlp_ablation_dlogp_median"] = [r3(x) for x in np.median(abl, 0)]
OUT["mlp_ablation_base_p"] = r3(np.mean(base_p))
print("MLP ablation dlogp", OUT["mlp_ablation_dlogp"], OUT["mlp_ablation_dlogp_median"])

# 4c. key-value memories (Geva et al. style): key = when a neuron fires; value = which tokens its output row promotes
acts = {l: [] for l in range(NL)}
hs = [blocks[l].mlp.act.register_forward_hook(lambda m, i, o, l=l: acts[l].append(o[0].half())) for l in range(NL)]
NCH = 250
for i in range(NCH): model(chunks[i:i + 1])
for h in hs: h.remove()
acts = {l: torch.cat(v, 0) for l, v in acts.items()}               # (NCH*CTX, 3072)
flat = chunks[:NCH].reshape(-1)
pos_ok = (torch.arange(flat.numel()) % CTX) < CTX - 1               # need a next token in the same chunk
K = 25
agree_rate = []; base_rate = []; examples = []
for l in range(NL):
    a = acts[l].float(); a[~pos_ok] = -1e9
    Wout = blocks[l].mlp.c_proj.weight                              # (3072, D)
    vals = (Wout @ W_U)                                             # (3072, V): value vector read through the unembedding
    vtop = vals.topk(10, -1).indices
    rng_n = torch.Generator().manual_seed(l)
    neurons = torch.randperm(3072, generator=rng_n)[:300]
    hits = 0
    for n in neurons.tolist():
        top_pos = a[:, n].topk(K).indices
        nxt = flat[top_pos + 1]
        if np.isin(nxt.numpy(), vtop[n].numpy()).any(): hits += 1
    agree_rate.append(hits / len(neurons))
    # chance: same neurons, but 25 random positions instead of the neuron's top triggers
    ok_idx = torch.nonzero(pos_ok)[:, 0]
    bh = 0
    for n in neurons.tolist():
        rp = ok_idx[torch.randint(0, ok_idx.numel(), (K,), generator=rng_n)]
        if np.isin(flat[rp + 1].numpy(), vtop[n].numpy()).any(): bh += 1
    base_rate.append(bh / len(neurons))
    if l >= 8:   # collect readable examples from upper layers: neurons whose value top token matches many triggers
        scores = []
        for n in range(3072):
            top_pos = a[:, n].topk(K).indices
            nxt = flat[top_pos + 1]
            m = int(np.isin(nxt.numpy(), vtop[n, :1].numpy()).sum())
            scores.append(m)
        best = np.argsort(scores)[::-1][:6]
        for n in best:
            n = int(n); top_pos = a[:, n].topk(8).indices
            trig = []
            for p_ in top_pos.tolist():
                s = max(0, p_ - 10)
                trig.append({"ctx": tok.decode(flat[s:p_ + 1]), "next": tok.decode([int(flat[p_ + 1])]), "act": r3(a[p_, n])})
            examples.append({"neuron": f"L{l}.N{n}", "layer": l, "matches_of_25": scores[n],
                             "value_top": [tok.decode([int(t)]) for t in vtop[n, :8]], "triggers": trig})
OUT["kv_agreement"] = [r3(x) for x in agree_rate]
OUT["kv_baseline_by_layer"] = [r3(x) for x in base_rate]
OUT["kv_baseline"] = r3(np.mean(base_rate))
OUT["kv_examples"] = examples
print("KV agreement by layer", OUT["kv_agreement"])
for e in examples[:12]: print(e["neuron"], e["matches_of_25"], e["value_top"][:5], [t["ctx"][-25:] + "|" + t["next"] for t in e["triggers"][:3]])

with open("book/assets/ch4_data.js", "w") as f:
    f.write("/* REAL measurements on GPT-2 small for Chapter 4. Generated by book/scripts/ch4_experiments.py. Do not edit by hand. */\n")
    f.write("window.CH4 = " + json.dumps(OUT, separators=(",", ":")) + ";\n")
print("wrote book/assets/ch4_data.js")
