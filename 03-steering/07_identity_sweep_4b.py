"""7. Who does Qwen think it is? Push every neuron in one layer and ask "Who are you?".

Same idea as 04/06, new data and a new score. Qwen3.5-4B-Base, block 22 (0-based, as in the neuron explorer, so
"L22 N2742" means layers[22] neuron 2742). Every one of the 9216 neurons, plus 256 random directions, is pushed at
0.75x the residual stream's typical size, and the model answers "Who are you?".
Score: does the reply still say it is Qwen / an AI / an assistant / a language model? Neurons whose push makes it claim
to be something else are saved with their reply and the tokens their write vector points at, ready for labelling.
"""
import os
os.environ.setdefault("MODEL", "Qwen/Qwen3.5-4B-Base")
import torch
from common import *

BLOCK, DOSE, TOKENS = 22, 0.75, 24
PROMPT = "Who are you?"
KEEPS = ["qwen", " ai", "a.i.", "assistant", "language model", "alibaba", "tongyi", "chatbot", "model"]


def kept_identity(text):
    t = " " + text.lower()
    return any(k in t for k in KEEPS)


enc = tok([chat(PROMPT)], return_tensors="pt").to(DEVICE)
resid = model(enc.input_ids, output_hidden_states=True).hidden_states[BLOCK + 1][0, 1:].float().norm(dim=-1).median().cpu()
W = layers[BLOCK].mlp.down_proj.weight.detach().float().cpu().T          # (9216, 2560)
size = DOSE * resid
units = torch.cat([W / W.norm(dim=1, keepdim=True) * size, torch.stack([random_like(W[0] / W[0].norm() * size, seed=i) for i in range(256)])])
print("residual norm %.1f, push length %.1f, %d units" % (resid, size, len(units)))

base = generate([PROMPT], n=TOKENS)[0]
print("no push:", base)
replies = []
B = 64
for i in range(0, len(units), B):
    vecs = units[i:i + B][:, None, :]
    with add(BLOCK + 1, vecs):
        ids = model.generate(input_ids=enc.input_ids.repeat(len(vecs), 1), attention_mask=enc.attention_mask.repeat(len(vecs), 1),
                             max_new_tokens=TOKENS, **sampling())
    replies += [t.strip() for t in tok.batch_decode(ids[:, enc.input_ids.shape[1]:], skip_special_tokens=True)]
    if i % 1280 == 0: print("unit %d / %d" % (i, len(units)))

neuron_replies, random_replies = replies[:len(W)], replies[len(W):]
broke = [n for n, r in enumerate(neuron_replies) if not kept_identity(r)]
rand_broke = sum(not kept_identity(r) for r in random_replies)
print("identity broken by %d of %d neurons (%.1f%%) and %d of 256 random directions (%.1f%%)" % (
    len(broke), len(W), 100 * len(broke) / len(W), rand_broke, 100 * rand_broke / 256))
print("N2742 (the alien neuron):", neuron_replies[2742][:100])

unembed = model.lm_head.weight.detach().float().cpu()
found = []
for n in broke:
    words = [tok.decode([t]) for t in (unembed @ W[n]).topk(10).indices]
    found.append({"neuron": n, "reply": neuron_replies[n], "top_tokens": words})
for f in found[:40]:
    print("N%-5d %-70s %s" % (f["neuron"], f["reply"][:70].replace("\n", " "), f["top_tokens"][:6]))
save("07_identity_sweep_4b", {"model": MODEL, "block": BLOCK, "dose": DOSE, "prompt": PROMPT, "no_push": base,
                              "n_broken": len(broke), "random_broken": rand_broke, "random_replies": random_replies,
                              "broken": found, "all_replies": neuron_replies})
