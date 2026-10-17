"""Is there a tomato neuron? Difference of means over every MLP neuron, for both models.

Each of the 24 sentences in data/tomato_sentences.py is filled once with "tomato" and once with each
of the other objects. For every block we read the MLP's hidden activation (the input of down_proj,
one value per neuron) at the last token of the filled-in word, and rank neurons by
    z = (mean over tomato - mean over others) / std over others.
A clean tomato neuron shows up as one z far above the rest.

For the top neurons we also read which tokens their down_proj column pushes up (through the final
norm and the unembedding), and for the 4B tomato neuron (block 8, neuron 4948) its activation on
every token of a few phrases.

Layer numbers count blocks from 1, like the rest of this folder. Forward passes only (no generation).
Writes out/08_tomato_neurons.json.
"""
import json, os, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from data.tomato_sentences import TEMPLATES, TARGET, OTHERS

torch.set_grad_enabled(False)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUT = os.path.join(os.path.dirname(__file__), "out")

PHRASES = ["I made tomato soup for dinner.", "She poured red sauce on the pizza.",
           "Pass the ketchup, please.", "He ate an apple after lunch.", "The chair was by the window."]


def load(name):
    tok = AutoTokenizer.from_pretrained(name)
    dtype = torch.float32 if "0.6B" in name else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(DEVICE).eval()
    return tok, model


def mlp_acts(model, tok, text, positions):
    """MLP hidden activation (input to down_proj) per block, at the given token positions."""
    grabbed = {}
    hooks = []
    for i, b in enumerate(model.model.layers):
        def hook(m, args, i=i):
            grabbed[i] = args[0][0].float().cpu()
        hooks.append(b.mlp.down_proj.register_forward_pre_hook(hook))
    ids = tok(text, return_tensors="pt").input_ids.to(DEVICE)
    model(ids)
    for h in hooks:
        h.remove()
    return torch.stack([grabbed[i][positions] for i in range(len(model.model.layers))])  # [blocks, pos, neurons]


def word_end(tok, template, word):
    """Token index of the last token of `word` inside the filled template."""
    pre = template.split("{}")[0] + word
    return len(tok(pre).input_ids) - 1


def top_tokens(model, tok, block, neuron, k=8):
    w = model.model.layers[block].mlp.down_proj.weight[:, neuron].float()
    logits = model.lm_head.weight.float() @ model.model.norm(w.to(model.model.norm.weight.dtype)).float()
    return [tok.decode([i]) for i in logits.topk(k).indices.tolist()]


def run(name):
    print(f"\n=== {name}: which MLP neurons stand out for 'tomato'? ===", flush=True)
    tok, model = load(name)
    tomato, others = [], []
    for t in TEMPLATES:
        tomato.append(mlp_acts(model, tok, t.format(TARGET), [word_end(tok, t, TARGET)])[:, 0])
        for o in OTHERS:
            others.append(mlp_acts(model, tok, t.format(o), [word_end(tok, t, o)])[:, 0])
    T, O = torch.stack(tomato), torch.stack(others)            # [n, blocks, neurons]
    z = (T.mean(0) - O.mean(0)) / (O.std(0) + 1e-3)             # [blocks, neurons]
    flat = z.flatten()
    order = flat.argsort(descending=True)
    nb = z.shape[1]
    top = []
    for idx in order[:12].tolist():
        b, n = divmod(idx, nb)
        top.append({"block": b + 1, "neuron": n, "z": round(z[b, n].item(), 2),
                    "tomato_mean": round(T[:, b, n].mean().item(), 3), "others_mean": round(O[:, b, n].mean().item(), 3),
                    "top_tokens": top_tokens(model, tok, b, n)})
        t = top[-1]
        print(f"  block {t['block']:>2} · neuron {t['neuron']:<5}  z {t['z']:>6.1f}   writes -> "
              + ", ".join(w.strip() or repr(w) for w in t["top_tokens"][:6]), flush=True)
    res = {"model": name, "blocks": z.shape[0], "neurons_per_block": nb,
           "z_sorted_top200": [round(v, 2) for v in flat[order[:200]].tolist()],
           "z_quantiles": {q: round(flat.quantile(q).item(), 2) for q in (0.5, 0.9, 0.99, 0.999)},
           "top": top}
    if "4B" in name:
        b, n = 8 - 1, 4948
        res["neuron_4948"] = {"z": round(z[b, n].item(), 2), "rank": int((flat > z[b, n]).sum().item()) + 1, "phrases": []}
        for p in PHRASES:
            ids = tok(p).input_ids
            a = mlp_acts(model, tok, p, list(range(len(ids))))[b, :, n]
            res["neuron_4948"]["phrases"].append({"text": p, "tokens": [tok.decode([i]) for i in ids],
                                                 "activation": [round(v, 3) for v in a.tolist()]})
        print("\n=== block 8 · neuron 4948: activation on each token ===", flush=True)
        for ph in res["neuron_4948"]["phrases"]:
            print("  " + "  ".join(f"{t.strip() or '_'}({a:+.1f})" for t, a in zip(ph["tokens"], ph["activation"])),
                  flush=True)
    del model
    if DEVICE == "mps":
        torch.mps.empty_cache()
    return res


if __name__ == "__main__":
    out = {"method": __doc__.split("\n")[0], "models": [run("Qwen/Qwen3-0.6B"), run("Qwen/Qwen3.5-4B-Base")]}
    with open(os.path.join(OUT, "08_tomato_neurons.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("\nwrote out/08_tomato_neurons.json")
