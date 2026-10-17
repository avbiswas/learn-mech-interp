"""3. Optimized removal (Heretic-style): let a search pick the edit's settings instead of choosing them by hand.

Each trial picks random settings: which layer's sad direction to use, how strongly to remove it (0 = not at all,
1 = fully), and the range of blocks to remove it from. It is scored on the tune topics:
  - mood when asked for "two sad sentences" (higher = removal worked)
  - KL divergence on 40 facts (lower = less damage)
Winner: the least damaging trial that still turns sad requests happy (mood above 0.8). It is then checked on the test topics.
Control: the same search with a random direction in place of the sad one.
Heretic uses a smarter optimizer (Optuna) and more knobs; plain random search keeps the idea visible.
"""
import random
import torch, matplotlib.pyplot as plt
from common import *
from data.topics import TUNE, TEST, prompts

saved = torch.load(OUT + "/sad_direction.pt")
all_dirs = saved["all_layers"]                                         # (29, 1024)
tune, test = prompts(TUNE, "sad"), prompts(TEST, "sad")
before = fact_logprobs()
TRIALS = 40


def score(direction, strength, which, ps):
    with remove(direction, strength, which):
        m = avg(mood(generate(ps)))
        return m, kl(before, fact_logprobs())


def search(pick_direction, seed):
    rng, trials = random.Random(seed), []
    for t in range(TRIALS):
        src = rng.randrange(4, 25)
        strength = rng.uniform(0.2, 1.2)
        first = rng.randrange(1, 21); last = rng.randrange(first + 1, 29)
        m, k = score(pick_direction(src), strength, range(first, last + 1), tune)
        trials.append({"direction_layer": src, "strength": r3(strength), "blocks": [first, last], "mood": r3(m), "kl": r3(k)})
        print("trial %2d: direction from layer %2d, strength %.2f, blocks %2d-%2d -> mood %+.2f, KL %.3f" % (t, src, strength, first, last, m, k))
    return trials


real = search(lambda L: all_dirs[L], seed=0)
rand = search(lambda L: random_like(all_dirs[L], seed=L), seed=0)

good = [t for t in real if t["mood"] > 0.8]
best = min(good, key=lambda t: t["kl"])
blocks = range(best["blocks"][0], best["blocks"][1] + 1)
test_mood, test_kl = score(all_dirs[best["direction_layer"]], best["strength"], blocks, test)
hand_mood, hand_kl = score(saved["direction"], 1.0, None, test)
print("best trial:", best)
print("test topics: optimized mood %+.2f KL %.3f | hand-picked (01/02) mood %+.2f KL %.3f" % (test_mood, test_kl, hand_mood, hand_kl))
print("random-direction search: best mood %+.2f" % max(t["mood"] for t in rand))

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.scatter([t["kl"] for t in rand], [t["mood"] for t in rand], color=MUTED, s=25, label="random direction (control)")
ax.scatter([t["kl"] for t in real], [t["mood"] for t in real], color=SERIES[1], s=25, label="sad direction")
ax.scatter([best["kl"]], [best["mood"]], color=SERIES[2], s=120, marker="*", zorder=3, label="chosen")
ax.axhline(0.8, color=GRID, ls="--")
ax.set(xlabel="damage (KL divergence on facts)", ylabel='mood when asked for "sad sentences"', ylim=(-1, 1),
       title="40 random settings: cheerful anyway, at the lowest damage")
ax.legend(frameon=False, fontsize=9)
save("03_optimized_removal", {"best": best, "test": {"optimized": {"mood": r3(test_mood), "kl": r3(test_kl)},
                                                     "hand_picked": {"mood": r3(hand_mood), "kl": r3(hand_kl)}},
                              "trials": real, "random_direction_trials": rand}, fig)
