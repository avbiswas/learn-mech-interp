"""3c. From the direction down to neurons: how few MLP neurons does it take to make the model sad?

Read: for every block from 8 to 20, each MLP neuron's average activation on the sad replies minus on the happy replies
(the same 32 pairs as 01, each read from where it differs from its pair, but inside the MLP instead of the residual stream).
Score: how much pushing that neuron moves the residual stream along the sad direction from 01
  = (its sad - happy difference) x (its write vector . sad direction)
Keep the top k neurons across all 13 blocks and push only those, each by c x its own difference.
Sweep k and c on the tune topics; pick the saddest setting that keeps facts within 0.1 of the unedited model.
Then on the test topics: compare against k random neurons from the same blocks, pushed just as hard.
"""
import random
from contextlib import ExitStack
import torch, matplotlib.pyplot as plt
from common import *
from data.mood_pairs import PAIRS
from data.topics import TUNE, TEST, prompts

BLOCKS = list(range(8, 21))
NEURONS = layers[0].mlp.down_proj.weight.shape[1]                     # 3072 per block in Qwen3-0.6B
H, S = [h for h, s in PAIRS], [s for h, s in PAIRS]
sad_dir = sad_directions()[SAD_LAYER]                                  # the sad direction of 01
r = sad_dir / sad_dir.norm()

delta, score = {}, {}
for L in BLOCKS:
    delta[L] = reply_neuron_acts(FEEL, S, L, others=H).mean(0) - reply_neuron_acts(FEEL, H, L, others=S).mean(0)
    score[L] = delta[L] * (r @ layers[L - 1].mlp.down_proj.weight.float().cpu())   # push along the sad direction
flat = torch.cat([score[L] for L in BLOCKS])                           # (13 x 3072,): one score per neuron


def pushes(chosen, c):
    """chosen: indices into the flat list of neurons. Returns {block: push for each of its neurons}."""
    mask = torch.zeros(len(flat)); mask[chosen] = 1
    return {L: c * delta[L] * mask[j * NEURONS:(j + 1) * NEURONS] for j, L in enumerate(BLOCKS)}


def write_size(p):
    """Total length of what the pushes add to the residual stream, summed over blocks."""
    return float(sum((layers[L - 1].mlp.down_proj.weight.float().cpu() @ v).norm() for L, v in p.items()))


def run(p, topics):
    with ExitStack() as stack:
        for L, v in p.items(): stack.enter_context(push_neurons(L, v))
        replies, lp = generate(prompts(topics), n=40), fact_logprobs()
    return {"mood": r3(avg(mood(replies))), "facts": facts_correct(lp), "kl": r3(kl(before, lp))}, replies


before = fact_logprobs()
base_facts = facts_correct(before)

# 1. sweep on the tune topics
sweep = {}
for k in [10, 30, 100, 300]:
    for c in [4, 8, 16]:
        sweep[(k, c)], replies = run(pushes(flat.topk(k).indices, c), TUNE)
        print("top %3d neurons x%2d: %s | %s" % (k, c, sweep[(k, c)], replies[0][:80].replace("\n", " ")), flush=True)
ok = [kc for kc in sweep if sweep[kc]["facts"] >= base_facts - 0.1]
K, C = min(ok, key=lambda kc: sweep[kc]["mood"])
print("chosen: top %d neurons, x%d" % (K, C))

# 2. who are they?
winners = []
for i in flat.topk(K).indices.tolist()[:10]:
    L, n = BLOCKS[i // NEURONS], i % NEURONS
    words = [tok.decode([t]) for t in model.lm_head.weight.detach().cpu().float().matmul(write_vector(L, n)).topk(8).indices]
    winners.append({"block": L, "neuron": n, "sad_minus_happy": r3(delta[L][n]), "score": r3(score[L][n]), "top_tokens": words})
    print("block %2d neuron %4d: sad-happy %+.2f, writes towards %s" % (L, n, delta[L][n], words))

# 3. test: chosen neurons vs random neurons pushed just as hard vs the sad direction from 01
chosen = pushes(flat.topk(K).indices, C)
rand = pushes(torch.tensor(random.Random(0).sample(range(len(flat)), K)), C)
rand = {L: v * write_size(chosen) / write_size(rand) for L, v in rand.items()}   # same total push size
d01 = SAD_STRENGTH * sad_dir
results, examples = {}, {}
for name, p in {"no push": {}, f"top {K} neurons": chosen, f"{K} random neurons": rand}.items():
    results[name], replies = run(p, TEST); examples[name] = replies[:4]
with add(SAD_LAYER, d01):
    replies, lp = generate(prompts(TEST), n=40), fact_logprobs()
results["sad direction (01)"] = {"mood": r3(avg(mood(replies))), "facts": facts_correct(lp), "kl": r3(kl(before, lp))}
examples["sad direction (01)"] = replies[:4]
for name in results: print("%-20s %s | %s" % (name, results[name], examples[name][0][:80].replace("\n", " ")))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for j, c in enumerate([4, 8, 16]):
    ks = [10, 30, 100, 300]
    ax[0].plot([sweep[(k, c)]["kl"] for k in ks], [sweep[(k, c)]["mood"] for k in ks], "o-", color=SERIES[j], label=f"push x{c}")
    for k in ks: ax[0].annotate(str(k), (sweep[(k, c)]["kl"], sweep[(k, c)]["mood"]), fontsize=8, color=INK2, xytext=(4, 4), textcoords="offset points")
ax[0].set(xscale="log", xlabel="damage (KL on facts, log scale)", ylabel="mood (joy - sadness)",
          title="Fewer neurons, less damage (labels = neurons kept)")
ax[0].legend(frameon=False, fontsize=9)
names = list(results)
ax[1].bar([n.replace(" ", "\n") for n in names], [results[n]["mood"] for n in names], color=[MUTED, SERIES[0], MUTED, SERIES[1]])
ax[1].axhline(0, color=GRID)
ax[1].set(ylabel="mood (joy - sadness)", ylim=(-1, 1), title="On new topics")
save("03c_sparse_neurons", {"temperature": TEMPERATURE, "blocks": BLOCKS, "sweep": {"top %d x%d" % kc: v for kc, v in sweep.items()}, "chosen": {"k": K, "c": C},
                            "winners": winners, "test": results, "examples": examples}, fig)
