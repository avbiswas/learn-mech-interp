"""C1 (+ control). Can a straight line read a concept from Qwen3-0.6B's residual stream, layer by layer?

Two concepts, two kinds of reader:
  - sentiment of movie reviews (rotten_tomatoes): mean of the residual stream over the review's tokens (skipping token 0)
  - the sport of 80 famous athletes, from "<name>" alone: residual stream at the athlete's last token
Readers: logistic regression (trained probe) and difference of means (no training: project onto mean_A - mean_B).
Control: the same probes on a randomly initialised Qwen3-0.6B (same architecture, no training). Whatever the
random model scores is what the probe can do by itself; only the gap above it is the model's knowledge.
"""
import random, torch, numpy as np, pyarrow.parquet as pq, matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM
from common import load, save, r3, SERIES, MUTED
from data.athletes import ATHLETES

name = "Qwen/Qwen3-0.6B"
tok, model = load(name)
torch.manual_seed(0)
rand_model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(name), dtype=torch.float32).eval()

RT = hf_hub_download("rotten_tomatoes", "train.parquet", repo_type="dataset",
                     revision="aa13bc287fa6fcab6daf52f0dfb9994269ffea28")
rows = pq.read_table(RT).to_pylist(); random.seed(0); rows = random.sample(rows, 600)
reviews, sent = [r["text"] for r in rows], np.array([r["label"] for r in rows])
people = [(p, s) for s, ps in ATHLETES.items() for p in ps]
sports = sorted(ATHLETES); sport_y = np.array([sports.index(s) for _, s in people])


def features(m, texts, pool):
    feats = []
    for t in texts:
        hs = m(tok(t, return_tensors="pt").input_ids, output_hidden_states=True).hidden_states
        feats.append(torch.stack([h[0, 1:].mean(0) if pool == "mean" else h[0, -1] for h in hs]))
    return torch.stack(feats).numpy()                                  # (N, layers+1, D)


def diff_means_acc(X, y, folds=5):
    """Nearest-class-mean along the mean-difference direction(s), cross-validated."""
    idx = np.random.RandomState(0).permutation(len(y)); accs = []
    for k in range(folds):
        te = idx[k::folds]; tr = np.setdiff1d(idx, te)
        means = np.stack([X[tr][y[tr] == c].mean(0) for c in np.unique(y)])
        pred = ((X[te][:, None] - means[None]) ** 2).sum(-1).argmin(-1)
        accs.append((pred == y[te]).mean())
    return float(np.mean(accs))


res, fig, ax = {}, *plt.subplots(1, 2, figsize=(12, 4.3))
for k, (task, texts, y, pool, chance) in enumerate([("sentiment", reviews, sent, "mean", 0.5),
                                                     ("sport", [p for p, _ in people], sport_y, "last", 0.25)]):
    res[task] = {}
    for c, (lab, m) in enumerate([("trained Qwen3-0.6B", model), ("random-init Qwen3-0.6B", rand_model)]):
        X = features(m, texts, pool)
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        lr = [cross_val_score(LogisticRegression(max_iter=2000, C=0.1), X[:, L], y, cv=5).mean() for L in range(X.shape[1])]
        dm = [diff_means_acc(X[:, L], y) for L in range(X.shape[1])]
        res[task][lab] = {"logistic": [r3(v) for v in lr], "diff_means": [r3(v) for v in dm]}
        ax[k].plot(lr, color=SERIES[c], lw=2, label=f"{lab}: logistic probe")
        ax[k].plot(dm, color=SERIES[c], lw=2, ls="--", label=f"{lab}: difference of means")
        print(f"{task:9s} {lab:24s} best logistic {max(lr):.2f} (layer {int(np.argmax(lr))}), best diff-of-means {max(dm):.2f}, layer 0 {lr[0]:.2f}")
    ax[k].axhline(chance, color=MUTED, ls=":", lw=1)
    ax[k].set(xlabel="layer (0 = embeddings)", ylabel="cross-validated accuracy", ylim=(0, 1.02),
              title="Sentiment of a movie review" if task == "sentiment" else "Sport of an athlete, from the name alone")
    ax[k].legend(frameon=False, fontsize=8)
save("07_probing", res, fig)
