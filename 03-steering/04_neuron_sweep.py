"""4. Neuron sweep with random controls: is there a single "sad neuron"?

Every MLP neuron writes one fixed vector into the residual stream (its column of down_proj). So each neuron is a
ready-made steering vector. At SAD_LAYER (or LAYER=...), we push every neuron's vector, and 256 random directions, all at the
same length (the length of the sad push from 01), on 4 tune prompts, and score the replies' mood.
A neuron counts as a "sad neuron" only if it makes replies sadder than every random direction did.
The best one is then tested on the test topics, against a random direction of the same length.
"""
import torch, matplotlib.pyplot as plt
from common import *
from data.topics import TUNE, TEST, prompts

LAYER = int(os.environ.get("LAYER", SAD_LAYER))                       # LAYER=14 uv run python 04_neuron_sweep.py
sad_dir = sad_directions()[LAYER]                                      # the sad direction of 01, at this layer
SIZE = (SAD_STRENGTH * sad_dir).norm()
W = layers[LAYER - 1].mlp.down_proj.weight.detach().cpu().T            # (3072, 1024): what each neuron writes
neurons = W / W.norm(dim=1, keepdim=True) * SIZE
randoms = torch.stack([random_like(neurons[0], seed=i) for i in range(256)])
units = torch.cat([neurons, randoms])                                  # (3328, 1024)

ps = prompts(TUNE[:4])
enc = tok([chat(p) for p in ps], return_tensors="pt", padding=True).to(DEVICE)
PER = 16                                                               # 16 units x 4 prompts = 64 replies per batch
base = avg(mood(generate(ps, n=40)))
shift = []
for i in range(0, len(units), PER):
    vecs = units[i:i + PER].repeat_interleave(len(ps), 0)[:, None, :]  # one push per reply
    with add(LAYER, vecs):
        ids = model.generate(input_ids=enc.input_ids.repeat(len(vecs) // len(ps), 1),
                             attention_mask=enc.attention_mask.repeat(len(vecs) // len(ps), 1), max_new_tokens=40, **sampling())
    texts = tok.batch_decode(ids[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    m = torch.tensor(mood(texts)).view(-1, len(ps)).mean(1) - base
    shift += m.tolist()
    if i % 640 == 0: print("unit %d / %d" % (i, len(units)))
shift = torch.tensor(shift)
n_shift, r_shift = shift[:3072], shift[3072:]
sad_neurons = (n_shift < r_shift.min()).nonzero().flatten().tolist()
order = n_shift.argsort().tolist()
print("baseline mood %.2f; random directions shift mood by %.2f to %+.2f" % (base, r_shift.min(), r_shift.max()))
print("neurons sadder than every random direction:", len(sad_neurons), sad_neurons[:20])
top = []
for n in order[:5]:
    words = [tok.decode([i]) for i in model.lm_head.weight.detach().cpu().matmul(W[n]).topk(8).indices]
    top.append({"neuron": n, "mood_shift": r3(n_shift[n]), "top_tokens": words})
    print("neuron %4d: shift %+.2f, writes towards %s" % (n, n_shift[n], words))

# test the best neuron on new topics
best = order[0]
test = prompts(TEST)
results, examples = {}, {}
for name, v in {"no push": torch.zeros(1024), "best sad neuron": neurons[best], "random direction": random_like(neurons[best], seed=999),
                "sad direction (01)": SAD_STRENGTH * sad_dir}.items():
    with add(LAYER, v):
        replies = generate(test)
    results[name], examples[name] = avg(mood(replies)), replies[:3]
    print("%-20s mood %+.2f | %s" % (name, results[name], replies[0][:90].replace("\n", " ")))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
bins = torch.linspace(float(shift.min()), float(shift.max()), 40)
ax[0].hist(r_shift, bins, color=MUTED, density=True, alpha=0.8, label="256 random directions")
ax[0].hist(n_shift, bins, color=SERIES[0], density=True, alpha=0.6, label="3072 neurons")
ax[0].axvline(float(r_shift.min()), color=SERIES[1], ls="--", label="saddest random direction")
ax[0].set(xlabel="change in mood when pushed", ylabel="share", title=f"Pushing every neuron in layer {LAYER}, against random pushes")
ax[0].legend(frameon=False, fontsize=9)
ax[1].bar([k.replace(" ", "\n", 1) for k in results], list(results.values()), color=[MUTED, SERIES[0], MUTED, SERIES[1]])
ax[1].axhline(0, color=GRID)
ax[1].set(ylabel="mood (joy - sadness)", ylim=(-1, 1), title=f"Neuron {best} on new topics")
save("04_neuron_sweep" + ("" if LAYER == SAD_LAYER else f"_layer{LAYER}"), {"layer": LAYER, "temperature": TEMPERATURE, "baseline_mood": r3(base), "random_shift_range": [r3(r_shift.min()), r3(r_shift.max())],
                         "sad_neurons": sad_neurons, "top5": top, "test": {k: r3(v) for k, v in results.items()},
                         "examples": examples, "neuron_shift": [r3(x) for x in n_shift]}, fig)
