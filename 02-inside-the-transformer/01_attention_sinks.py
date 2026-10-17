"""B1a. How many heads park their attention on the first token, in three models, and what do they read there?

Measures, averaged over 20 ordinary sentences:
  1. share of attention each head sends to position 0 (query positions 1..end)
  2. norm of the value vector at position 0 vs other positions (a sink should read almost nothing)
  3. does the sink follow the *token* or the *position*? Prepend different first tokens and re-measure.
"""
import torch, matplotlib.pyplot as plt
from common import load, blocks, save, r3, SERIES, MUTED
from data.texts import SENTENCES

MODELS = ["gpt2", "HuggingFaceTB/SmolLM2-135M", "Qwen/Qwen3-0.6B"]


def value_hooks(model, store):
    hs = []
    for L, b in enumerate(blocks(model)):
        if hasattr(b, "attn"):                       # GPT-2: c_attn returns [q | k | v]
            D = model.config.n_embd
            hs.append(b.attn.c_attn.register_forward_hook(lambda m, i, o, L=L: store.__setitem__(L, o[0, :, 2 * D:])))
        else:
            hs.append(b.self_attn.v_proj.register_forward_hook(lambda m, i, o, L=L: store.__setitem__(L, o[0])))
    return hs


def measure(tok, model, texts, prefix=""):
    sink, vratio = 0, 0
    for t in texts:
        ids = tok(prefix + t, return_tensors="pt").input_ids
        store = {}
        hs = value_hooks(model, store)
        att = model(ids, output_attentions=True).attentions
        for h in hs: h.remove()
        sink = sink + torch.stack([a[0, :, 1:, 0].mean(-1) for a in att])              # (L, H)
        vn = torch.stack([store[L].norm(dim=-1) for L in range(len(att))])            # (L, T)
        vratio = vratio + vn[:, 0] / vn[:, 1:].mean(-1)
    return sink / len(texts), vratio / len(texts)


res = {}
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for c, name in enumerate(MODELS):
    tok, model = load(name, eager=True)
    sink, vratio = measure(tok, model, SENTENCES)
    L, H = sink.shape
    depth = torch.arange(L) / (L - 1)
    frac = (sink > 0.5).float().mean(-1)
    ax[0].plot(depth, frac, color=SERIES[c], lw=2, label=f"{name.split('/')[-1]} ({(sink > 0.5).sum().item()}/{L*H} heads)")
    ax[1].plot(depth, vratio, color=SERIES[c], lw=2, label=name.split("/")[-1])
    # Swap the first token: is the sink about position 0, or about the word?
    swaps = {}
    for pre in ["Hello. ", "\n", "Banana "]:
        s2, _ = measure(tok, model, SENTENCES, prefix=pre)
        swaps[repr(pre)] = r3((s2 > 0.5).float().mean())
    res[name] = {"layers": L, "heads": H, "sink_heads_over_0.5": int((sink > 0.5).sum()), "total_heads": L * H,
                 "share_by_layer": [r3(x) for x in frac], "value_norm_pos0_over_rest_by_layer": [r3(x) for x in vratio],
                 "share_over_0.5_with_prefix": swaps, "sink_matrix": [[r3(x) for x in r] for r in sink.tolist()]}
    print(name, res[name]["sink_heads_over_0.5"], "/", L * H, "| v-norm ratio median", r3(vratio.median()), "| prefixes", swaps)

ax[0].set(xlabel="depth (0 = first layer, 1 = last)", ylabel="share of heads with >50% attention on token 0", title="Most heads park on the first token", ylim=(0, 1.02))
ax[1].axhline(1, color=MUTED, ls="--", lw=1)
ax[1].set(xlabel="depth", ylabel="value norm at token 0 ÷ other tokens", title="…and read almost nothing there", yscale="log")
for a in ax: a.legend(frameon=False, fontsize=9)
save("01_attention_sinks", res, fig)
