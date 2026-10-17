"""3b. Where a search earns its keep: make the model bring up tomatoes in every answer, without losing the plot.

Tomato direction: 24 sentences with a blank, filled with "tomato" minus the same sentences filled with 12 other things
(data/tomato_sentences.py). Then we add it while the model answers "Write two sentences about <topic>."
A reply counts as a success only if all three hold:
  - it mentions tomatoes
  - it still mentions the topic it was asked about (asked about a dog, it talks about the dog)
  - it isn't stuck repeating itself (at least half its words are different)
We push only while the model writes its answer, not while it reads the question; pushing on the question makes it
forget what it was asked. Too little push: no tomatoes. Too much: "tomatoes tomatoes tomatoes". The good window is narrow and moves with the layer.
  1. By hand: the round numbers a person would try, 5 layers x 4 strengths.
  2. Search: 60 random settings, with finer strengths and one more knob (spread the push over 1-5 blocks).
Both are chosen on 36 topics (train + tune) and compared on the 24 test topics.
"""
import random, re
import torch, matplotlib.pyplot as plt
from common import *
from data.tomato_sentences import TEMPLATES, TARGET, OTHERS
from data.topics import TRAIN, TUNE, TEST, prompts

CHOOSE = TRAIN + TUNE                                                 # 36 topics to pick settings on
N = len(layers)                                                        # 28 blocks in Qwen3-0.6B, 32 in Qwen3.5-4B
GRID_LAYERS = [round(N * f) for f in (0.2, 0.35, 0.5, 0.65, 0.8)]      # 0.6B: 6, 10, 14, 18, 22
TAG = "" if MODEL == "Qwen/Qwen3-0.6B" else "_" + MODEL.split("-")[-2].lower()   # Qwen3.5-4B-Base -> "_4b"


def mean_acts(texts):
    out = []
    for t in texts:
        hs = model(tok(t, return_tensors="pt").input_ids.to(DEVICE), output_hidden_states=True).hidden_states
        out.append(torch.stack([h[0].mean(0) for h in hs]).float().cpu())
    return torch.stack(out).mean(0)                                    # (blocks + 1, width)


fill = lambda word: [t.format(word) for t in TEMPLATES]
d = mean_acts(fill(TARGET)) - torch.stack([mean_acts(fill(w)) for w in OTHERS]).mean(0)   # one tomato direction per layer


def success(reply, topic):
    r = reply.lower()
    tomato = "tomato" in r or "番茄" in r
    on_topic = topic.split()[-1].lower().rstrip("s") in r
    words = re.findall(r"[a-z]+", r)                                   # letters only, so "tomato.." counts as "tomato"
    varied = len(set(words)) >= 0.5 * max(1, len(words))
    return tomato and on_topic and varied


def run(topics, layer, strength, spread=1):
    """Add strength x the tomato direction, split evenly over `spread` blocks starting at `layer`, on new tokens only."""
    v = (strength / spread * d[layer]).to(DEVICE)
    push = lambda m, i, o: o + v if o.shape[1] == 1 else o            # while generating, each step is one new token
    hooks = [layers[l - 1].register_forward_hook(push) for l in range(layer, min(layer + spread, N + 1))]
    replies = generate(prompts(topics), n=50)
    [h.remove() for h in hooks]
    return avg([success(r, t) for r, t in zip(replies, topics)]), replies


# 1. by hand: round numbers
grid = {(L, s): run(CHOOSE, L, s)[0] for L in GRID_LAYERS for s in [5, 10, 15, 20]}
hand = max(grid, key=grid.get)
print("by hand, success on the 36 choosing topics:", {"L%d x%d" % k: r3(v) for k, v in grid.items()}, "-> best", hand)

# 2. random search
rng, trials = random.Random(0), []
for t in range(60):
    L, s, k = rng.randrange(round(N * 0.15), round(N * 0.9)), rng.uniform(4, 20), rng.randrange(1, 6)
    score, replies = run(CHOOSE, L, s, k)
    trials.append({"layer": L, "strength": r3(s), "spread": k, "success": r3(score)})
    print("trial %2d: layer %2d, strength %5.2f, spread over %d blocks -> success %.2f | %s" % (
        t, L, s, k, score, replies[0][:70].replace("\n", " ")))
best = max(trials, key=lambda t: t["success"])
print("search best:", best)

# compare on the test topics
results, examples = {}, {}
for name, (L, s, k) in {"by hand": (hand[0], hand[1], 1), "search": (best["layer"], best["strength"], best["spread"])}.items():
    results[name], replies = run(TEST, L, s, k)
    examples[name] = dict(zip(TEST, replies))
    print("%-8s (layer %d, x%.1f, %d blocks) test success %.2f" % (name, L, s, k, results[name]))
    for r in replies[:4]: print("    ", r[:120].replace("\n", " "))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
Ls, Ss = GRID_LAYERS, [5, 10, 15, 20]
im = ax[0].imshow([[grid[(L, s)] for L in Ls] for s in Ss], origin="lower", cmap="viridis", vmin=0, vmax=1, aspect="auto",
                  extent=(Ls[0] - (Ls[1] - Ls[0]) / 2, Ls[-1] + (Ls[1] - Ls[0]) / 2, 2.5, 22.5))
ax[0].scatter([t["layer"] for t in trials], [t["strength"] for t in trials], c=[t["success"] for t in trials], cmap="viridis",
              vmin=0, vmax=1, edgecolors="white", s=40)
ax[0].scatter([best["layer"]], [best["strength"]], marker="*", s=300, color=SERIES[1], edgecolors="white", label="best found by search")
ax[0].set(xlabel="layer", ylabel="strength", title="Share of replies that are about the topic AND tomatoes")
ax[0].legend(frameon=False, fontsize=9, loc="upper right"); fig.colorbar(im, ax=ax[0])
ax[1].bar(list(results), list(results.values()), color=[MUTED, SERIES[1]])
ax[1].set(ylim=(0, 1), ylabel="success on new topics", title="Round numbers vs search")
save("03b_tomato_search" + TAG, {"model": MODEL, "grid": {"L%d x%d" % k: r3(v) for k, v in grid.items()}, "hand": list(hand), "best": best,
                           "test": {k: r3(v) for k, v in results.items()}, "trials": trials, "examples": examples}, fig)
