"""5. A trained steering vector (ReFT-style): learn the push with gradient descent instead of subtracting averages.

Training data: the model's own replies to "Write two sad sentences about <topic>" (24 training topics), shown to the
model as replies to the plain "Write two sentences about <topic>". So the vector has to supply the sadness itself.
We freeze the model and train only one vector (1024 numbers) added after SAD_LAYER, to make those sad replies likely.
Test topics: mood with the trained vector, against a random vector of the same length, the sad direction from 01,
and simply asking for sad sentences. Plus the cosine similarity between the trained vector and the sad direction.
"""
import torch, matplotlib.pyplot as plt
from common import *
from data.topics import TRAIN, TEST, prompts

LAYER = SAD_LAYER
d = sad_directions()[LAYER]                                            # the sad direction of 01, computed fresh
plain, targets = prompts(TRAIN), generate(prompts(TRAIN, "sad"))
keep = [i for i, m in enumerate(mood(targets)) if m < 0]               # the small model doesn't always comply: keep the sad ones
plain, targets = [plain[i] for i in keep], [targets[i] for i in keep]
print("kept %d of %d replies as sad" % (len(keep), len(TRAIN)))
print("training example: %s -> %s" % (plain[0], targets[0][:100].replace("\n", " ")))

data = []
for p, r in zip(plain, targets):
    ids = tok(chat(p, r), return_tensors="pt").input_ids.to(DEVICE)
    labels = ids.clone(); labels[:, :len(tok(chat(p)).input_ids)] = -100  # only the reply counts
    data.append((ids, labels))

model.requires_grad_(False)
v = torch.zeros(1024, device=DEVICE, requires_grad=True)
opt = torch.optim.Adam([v], lr=0.02)
losses = []
with torch.enable_grad(), add(LAYER, v):
    for step in range(100):
        opt.zero_grad()
        loss = sum(model(ids, labels=labels).loss for ids, labels in data) / len(data)
        loss.backward(); opt.step()
        losses.append(loss.item())
        if step % 20 == 0: print("step %3d loss %.3f |v| %.1f" % (step, loss.item(), v.detach().norm()))
v = v.detach().cpu()
cos = float(torch.nn.functional.cosine_similarity(v, d, dim=0))
print("|trained vector| %.1f, |sad push| %.1f, cosine with sad direction %.2f" % (v.norm(), (SAD_STRENGTH * d).norm(), cos))

test = prompts(TEST)
before = fact_logprobs()
results, examples = {}, {}
for name, vec in {"no push": torch.zeros(1024), "trained vector": v, "random vector": random_like(v), "sad direction (01)": SAD_STRENGTH * d}.items():
    with add(LAYER, vec):
        replies = generate(test)
        lp = fact_logprobs()
    results[name] = {"mood": avg(mood(replies)), "facts": facts_correct(lp), "kl": kl(before, lp)}; examples[name] = replies[:3]
asked = generate(prompts(TEST, "sad"))
results["just ask for sad sentences"] = {"mood": avg(mood(asked)), "facts": facts_correct(before), "kl": 0.0}; examples["just ask for sad sentences"] = asked[:3]
for k, r in results.items():
    print("%-28s mood %+.2f  facts %.2f  KL %.3f | %s" % (k, r["mood"], r["facts"], r["kl"], examples[k][0][:80].replace("\n", " ")))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].plot(losses, color=SERIES[0]); ax[0].set(xlabel="step", ylabel="loss on sad replies", title="Training one vector, model frozen")
ax[1].bar([k.replace(" ", "\n", 1) for k in results], [r["mood"] for r in results.values()],
          color=[MUTED, SERIES[0], MUTED, SERIES[1], SERIES[2]])
ax[1].axhline(0, color=GRID)
ax[1].set(ylabel="mood (joy - sadness)", ylim=(-1, 1), title=f"Trained vector on new topics (cosine with sad direction {cos:.2f})")
save("05_trained_vector", {"layer": LAYER, "temperature": TEMPERATURE, "cosine_with_sad_direction": r3(cos), "vector_norm": r3(v.norm()),
                           "results": {k: {m: r3(x) for m, x in r.items()} for k, r in results.items()},
                           "losses": [r3(x) for x in losses], "examples": examples}, fig)
