"""Real activation patching on GPT-2 small for the IOI prompt (denoising: clean -> corrupted)."""
import json, torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

torch.set_grad_enabled(False)
tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()
H, D = 12, 64

pairs = [
    ("When Mary and John went to the store, John gave a drink to", "When Mary and John went to the store, Mary gave a drink to", " Mary", " John"),
    ("When Mary and John went to the store, John gave a drink to", "When Alice and Bob went to the store, Bob gave a drink to", " Mary", " John"),
]

def run(ids, hooks=()):
    hs = [m.register_forward_pre_hook(f) if kind == "pre" else m.register_forward_hook(f) for kind, m, f in hooks]
    out = model(ids, output_hidden_states=True)
    for h in hs: h.remove()
    return out

def capture(ids):
    cache = {"attn": {}, "mlp": {}, "z": {}}
    hooks = []
    for L, blk in enumerate(model.transformer.h):
        hooks.append(("post", blk.attn, lambda m, i, o, L=L: cache["attn"].__setitem__(L, o[0].clone())))
        hooks.append(("post", blk.mlp, lambda m, i, o, L=L: cache["mlp"].__setitem__(L, o.clone())))
        hooks.append(("pre", blk.attn.c_proj, lambda m, i, L=L: cache["z"].__setitem__(L, i[0].clone())))
    out = run(ids, hooks)
    return out, cache

results = []
for clean, corr, a, b in pairs:
    ci = tok(clean, return_tensors="pt").input_ids
    xi = tok(corr, return_tensors="pt").input_ids
    assert ci.shape == xi.shape
    ta, tb = tok(a).input_ids[0], tok(b).input_ids[0]
    def ld(out): l = out.logits[0, -1]; return (l[ta] - l[tb]).item()
    cout, cc = capture(ci)
    xout = run(xi)
    clean_ld, corr_ld = ld(cout), ld(xout)
    norm = lambda v: (v - corr_ld) / (clean_ld - corr_ld)
    T = ci.shape[1]
    clean_hs = cout.hidden_states  # tuple len 13: resid_pre for layers 0..11, final
    resid = []
    for L in range(12):
        row = []
        for p in range(T):
            def f(m, args, L=L, p=p):
                x = args[0].clone(); x[:, p] = clean_hs[L][:, p]; return (x,) + args[1:]
            row.append(round(norm(ld(run(xi, [("pre", model.transformer.h[L], f)]))), 3))
        resid.append(row)
    attn, mlp = [], []
    for L in range(12):
        ra, rm = [], []
        for p in range(T):
            def fa(m, i, o, L=L, p=p):
                x = o[0].clone(); x[:, p] = cc["attn"][L][:, p]; return (x,) + tuple(o[1:])
            def fm(m, i, o, L=L, p=p):
                x = o.clone(); x[:, p] = cc["mlp"][L][:, p]; return x
            ra.append(round(norm(ld(run(xi, [("post", model.transformer.h[L].attn, fa)]))), 3))
            rm.append(round(norm(ld(run(xi, [("post", model.transformer.h[L].mlp, fm)]))), 3))
        attn.append(ra); mlp.append(rm)
    heads = []  # patch each head's output at ALL positions
    for L in range(12):
        row = []
        for hd in range(H):
            def fz(m, args, L=L, hd=hd):
                x = args[0].clone(); x[:, :, hd*D:(hd+1)*D] = cc["z"][L][:, :, hd*D:(hd+1)*D]; return (x,)
            row.append(round(norm(ld(run(xi, [("pre", model.transformer.h[L].attn.c_proj, fz)]))), 3))
        heads.append(row)
    heads_final = []
    for L in range(12):
        row = []
        for hd in range(H):
            def fz(m, args, L=L, hd=hd):
                x = args[0].clone(); x[:, -1, hd*D:(hd+1)*D] = cc["z"][L][:, -1, hd*D:(hd+1)*D]; return (x,)
            row.append(round(norm(ld(run(xi, [("pre", model.transformer.h[L].attn.c_proj, fz)]))), 3))
        heads_final.append(row)
    # attention pattern of key heads at final position (clean run)
    out = model(ci, output_attentions=True)
    pats = {f"{L}.{h}": [round(v, 3) for v in out.attentions[L][0, h, -1].tolist()] for L, h in [(9, 9), (9, 6), (10, 0), (7, 3), (8, 6), (3, 0), (5, 5), (10, 7), (11, 10)]}
    results.append(dict(clean=clean, corrupted=corr, answer=a, wrong=b,
        tokens=[tok.decode([t]) for t in ci[0]], corr_tokens=[tok.decode([t]) for t in xi[0]],
        clean_logit_diff=round(clean_ld, 3), corrupted_logit_diff=round(corr_ld, 3),
        resid_pre=resid, attn_out=attn, mlp_out=mlp, heads_all_pos=heads, heads_final_pos=heads_final, attn_patterns_final=pats))

json.dump(results, open("ioi_patching.json", "w"))
for r in results:
  print("LD", r["clean_logit_diff"], r["corrupted_logit_diff"])
  for key in ["heads_all_pos","heads_final_pos"]:
    top = sorted(((v, f"{L}.{h}") for L, row in enumerate(r[key]) for h, v in enumerate(row)), key=lambda t: -abs(t[0]))[:12]
    print(key, top)
r = results[1]
print("clean LD", r["clean_logit_diff"], "corr LD", r["corrupted_logit_diff"])
print(r["tokens"])
for L, row in enumerate(r["resid_pre"]): print(L, " ".join(f"{v:5.2f}" for v in row))
top = sorted(((v, f"{L}.{h}") for L, row in enumerate(r["heads_all_pos"]) for h, v in enumerate(row)), key=lambda t: -abs(t[0]))[:12]
print(top)
