"""1. The sad direction: difference of means, then add it to make the model sad.

Direction: 32 sentence pairs that say the same thing happily and sadly (data/mood_pairs.py), read as the model's own reply
to "Tell me how you feel.". Each sentence is averaged only from the token where it parts ways with its pair
("...I can't stop | smiling." vs "...I can't stop | crying."), so the shared words don't blur the difference.
At every layer: mean over sad sentences minus mean over happy sentences.
Layer: add SAD_STRENGTH (2x) the direction at each layer to see where it works; we steer at SAD_LAYER (12 = model.model.layers[11]).
Replies are sampled (TEMPERATURE in common.py, 0 = greedy).
Then, on the test prompts ("Write two sentences about a dog."): mood score at several strengths (negative = add happiness),
against two controls of the same length: a random direction, and a direction from shuffled happy/sad labels.
"""
import random
import torch, matplotlib.pyplot as plt
from common import *
from data.mood_pairs import PAIRS
from data.topics import TUNE, TEST, prompts

H, S = [h for h, s in PAIRS], [s for h, s in PAIRS]
happy = reply_acts(FEEL, H, others=S)                                  # (32, 29, 1024), each from where it differs from its pair
sad = reply_acts(FEEL, S, others=H)
d = sad.mean(0) - happy.mean(0)                                        # (29, 1024): one sad direction per layer (= sad_directions())

# where does it work? push each layer's direction 2x and see which makes the model saddest
STRENGTH = SAD_STRENGTH
tune = prompts(TUNE)
sweep = {}
for L in range(2, 27, 2):
    with add(L, STRENGTH * d[L]):
        sweep[L] = avg(mood(generate(tune)))
LAYER = SAD_LAYER
print("mood when pushing %dx at each layer" % STRENGTH + ":", {k: r3(v) for k, v in sweep.items()},
      "| saddest:", min(sweep, key=sweep.get), "| steering at", LAYER)

# controls: same length, same layer
push = STRENGTH * d[LAYER]
flip = [random.Random(i).random() < 0.5 for i in range(len(PAIRS))]
shuffled = torch.stack([(happy[i] - sad[i]) if f else (sad[i] - happy[i]) for i, f in enumerate(flip)]).mean(0)[LAYER]
controls = {"random direction": random_like(push), "shuffled labels": shuffled / shuffled.norm() * push.norm()}

test = prompts(TEST)
before = fact_logprobs()
curve, examples = {}, {}
for s in [-3, -2, -1, 0, 1, 2, 3, 4]:
    with add(LAYER, s * d[LAYER]):
        replies = generate(test)
    curve[s], examples[s] = avg(mood(replies)), replies[:3]
    print("strength %+d: mood %.2f | %s" % (s, curve[s], replies[0][:100].replace("\n", " ")))
ctrl = {}
for name, v in controls.items():
    with add(LAYER, v):
        replies = generate(test)
    ctrl[name] = avg(mood(replies)); examples[name] = replies[:3]
    print("%s (same length as the push): mood %.2f" % (name, ctrl[name]))
with add(LAYER, push):
    after = fact_logprobs()
damage = {"facts_before": facts_correct(before), "facts_after": facts_correct(after), "kl": kl(before, after)}
print("damage at %dx:" % STRENGTH + " facts %.2f -> %.2f, KL %.3f" % (damage["facts_before"], damage["facts_after"], damage["kl"]))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].plot(list(sweep), list(sweep.values()), "o-", color=SERIES[0])
ax[0].set(xlabel="layer", ylabel="mood (joy - sadness)", title="Where does pushing the sad direction work best?")
ax[1].plot(list(curve), list(curve.values()), "o-", color=SERIES[1], label="sad direction")
for (name, v), c in zip(ctrl.items(), [MUTED, SERIES[3]]):
    ax[1].scatter([STRENGTH], [v], color=c, s=60, zorder=3, label=name + " (same length as %dx)" % STRENGTH)
ax[1].axhline(0, color=GRID)
ax[1].set(xlabel="strength (negative = push towards happy)", ylabel="mood (joy - sadness)", ylim=(-1, 1),
          title=f"Add it at layer {LAYER} and the model turns sad")
ax[1].legend(frameon=False, fontsize=9)
save("01_sad_direction", {"layer": LAYER, "temperature": TEMPERATURE, "layer_sweep": {k: r3(v) for k, v in sweep.items()},
                          "strength_curve": {k: r3(v) for k, v in curve.items()}, "controls": {k: r3(v) for k, v in ctrl.items()},
                          "damage_at_push": {k: r3(v) for k, v in damage.items()}, "examples": {str(k): v for k, v in examples.items()}}, fig)
