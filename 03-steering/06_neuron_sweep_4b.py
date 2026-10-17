"""6. Is there a sad neuron in a bigger model? The neuron sweep of 04, on Qwen3.5-4B-Base (32 blocks, 9216 neurons each).

Generation is slow for this model on a Mac, so instead of pushing all 9216 neurons we narrow down first:
  1. Sad direction on this model, as in 01 (same sentence pairs), and the layer where pushing it works best.
  2. The 128 neurons whose write vector (column of down_proj) points most along the sad direction.
  3. Push each of them, 128 random neurons and 128 random directions, all at the same length, on 4 tune prompts.
A neuron counts only if it makes replies sadder than every random direction. Read the replies of the top ones,
then test the best neuron on the test topics.
"""
import os, random
os.environ.setdefault("MODEL", "Qwen/Qwen3.5-4B-Base")
import torch, matplotlib.pyplot as plt
from common import *
from data.mood_pairs import PAIRS
from data.topics import TUNE, TEST, prompts

Q = "Tell me how you feel."
d = reply_acts(Q, [s for h, s in PAIRS]).mean(0) - reply_acts(Q, [h for h, s in PAIRS]).mean(0)   # (33, 2560)
STRENGTH = 4
tune = prompts(TUNE)
sweep = {}
for L in [8, 12, 16, 20, 24]:
    with add(L, STRENGTH * d[L]):
        replies = generate(tune, n=40)
    sweep[L] = avg(mood(replies))
    print("layer %d: mood %+.2f | %s" % (L, sweep[L], replies[0][:90].replace("\n", " ")))
LAYER = min(sweep, key=sweep.get)
push = STRENGTH * d[LAYER]
print("-> layer", LAYER)

# narrow down: neurons whose write vector points along the sad direction
W = layers[LAYER - 1].mlp.down_proj.weight.detach().float().cpu().T     # (9216, 2560)
cos = (W / W.norm(dim=1, keepdim=True)) @ (push / push.norm())
top = cos.argsort(descending=True)[:128].tolist()
rand_neurons = random.Random(0).sample([n for n in range(len(W)) if n not in top], 128)
scaled = lambda n: W[n] / W[n].norm() * push.norm()
units = torch.stack([scaled(n) for n in top + rand_neurons] + [random_like(push, seed=i) for i in range(128)])
print("cosine with the sad direction: best neuron %.2f, 128th %.2f" % (cos[top[0]], cos[top[-1]]))

ps = prompts(TUNE[:4])
enc = tok([chat(p) for p in ps], return_tensors="pt", padding=True).to(DEVICE)
PER = 16
base = avg(mood(generate(ps, n=30)))
shift, texts_all = [], []
for i in range(0, len(units), PER):
    vecs = units[i:i + PER].repeat_interleave(len(ps), 0)[:, None, :]
    with add(LAYER, vecs):
        ids = model.generate(input_ids=enc.input_ids.repeat(len(vecs) // len(ps), 1),
                             attention_mask=enc.attention_mask.repeat(len(vecs) // len(ps), 1), max_new_tokens=30, do_sample=False)
    texts = tok.batch_decode(ids[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    shift += (torch.tensor(mood(texts)).view(-1, len(ps)).mean(1) - base).tolist()
    texts_all += [texts[j:j + len(ps)] for j in range(0, len(texts), len(ps))]
    print("unit %d / %d" % (i + PER, len(units)))
shift = torch.tensor(shift)
s_top, s_randn, s_rand = shift[:128], shift[128:256], shift[256:]
sad_neurons = [top[i] for i in range(128) if s_top[i] < s_rand.min()]
print("baseline mood %.2f; random directions shift %.2f to %+.2f; random neurons %.2f to %+.2f" % (
    base, s_rand.min(), s_rand.max(), s_randn.min(), s_randn.max()))
print("aligned neurons sadder than every random direction: %d of 128" % len(sad_neurons))
best5 = []
for i in s_top.argsort()[:5].tolist():
    n = top[i]
    words = [tok.decode([t]) for t in (model.lm_head.weight.detach().float().cpu() @ W[n]).topk(10).indices]
    best5.append({"neuron": n, "cosine": r3(cos[n]), "mood_shift": r3(s_top[i]), "top_tokens": words, "replies": texts_all[i]})
    print("neuron %4d (cos %.2f): shift %+.2f, writes towards %s\n   %s" % (n, cos[n], s_top[i], words, texts_all[i][0][:120].replace("\n", " ")))

best = best5[0]["neuron"]
test = prompts(TEST)
results, examples = {}, {}
for name, v in {"no push": torch.zeros_like(push), "best neuron": scaled(best), "random direction": random_like(push, seed=999),
                "sad direction": push}.items():
    with add(LAYER, v):
        replies = generate(test, n=40)
    results[name], examples[name] = avg(mood(replies)), replies[:4]
    print("%-18s mood %+.2f | %s" % (name, results[name], replies[0][:100].replace("\n", " ")))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
bins = torch.linspace(float(shift.min()), float(shift.max()), 30)
ax[0].hist(s_rand, bins, color=MUTED, alpha=0.8, label="128 random directions")
ax[0].hist(s_randn, bins, color=SERIES[3], alpha=0.6, label="128 random neurons")
ax[0].hist(s_top, bins, color=SERIES[0], alpha=0.6, label="128 neurons most aligned with sad")
ax[0].axvline(float(s_rand.min()), color=SERIES[1], ls="--", label="saddest random direction")
ax[0].set(xlabel="change in mood when pushed", ylabel="count", title=f"Qwen3.5-4B, layer {LAYER}")
ax[0].legend(frameon=False, fontsize=9)
ax[1].bar([k.replace(" ", "\n", 1) for k in results], list(results.values()), color=[MUTED, SERIES[0], MUTED, SERIES[1]])
ax[1].axhline(0, color=GRID)
ax[1].set(ylabel="mood (joy - sadness)", ylim=(-1, 1), title=f"Neuron {best} on new topics")
save("06_neuron_sweep_4b", {"model": MODEL, "layer": LAYER, "layer_sweep": {k: r3(v) for k, v in sweep.items()},
                            "baseline_mood": r3(base), "random_direction_shift_range": [r3(s_rand.min()), r3(s_rand.max())],
                            "random_neuron_shift_range": [r3(s_randn.min()), r3(s_randn.max())], "sad_neurons": sad_neurons,
                            "top5": best5, "test": {k: r3(v) for k, v in results.items()}, "examples": examples}, fig)
