"""C4. Is there a map of the world inside the model? (after Gurnee & Tegmark 2023, who used Llama-2 7B-70B)

For ~105 cities, take the residual stream at the last token of the city name, and fit a linear (ridge) regression to
latitude and longitude, cross-validated (each city is predicted by a probe that never saw it).
Compared across layers, and against a randomly initialised model of the same shape.
"""
import torch, numpy as np, matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict
from transformers import AutoConfig, AutoModelForCausalLM
from common import load, save, r3, SERIES, MUTED
from data.cities import CITIES

name = "Qwen/Qwen3-0.6B"
tok, model = load(name)
torch.manual_seed(0)
rand_model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(name), dtype=torch.float32).eval()
Y = np.array([[lat, lon] for _, lat, lon in CITIES])


def feats(m):
    return torch.stack([torch.stack(m(tok(c, return_tensors="pt").input_ids, output_hidden_states=True).hidden_states)[:, 0, -1]
                        for c, _, _ in CITIES]).numpy()                         # (N, L+1, D)


def fit(X):
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    pred = cross_val_predict(RidgeCV(alphas=np.logspace(0, 4, 9)), X, Y, cv=10)
    r2 = 1 - ((pred - Y) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
    return pred, r2


res = {}
for lab, m in [("trained", model), ("random-init", rand_model)]:
    X = feats(m)
    scores = [fit(X[:, L])[1] for L in range(X.shape[1])]
    best = int(np.argmax([s.mean() for s in scores]))
    pred, r2 = fit(X[:, best])
    err_km = 111 * np.sqrt(((pred - Y) ** 2 * np.array([1, 0.6])).sum(-1))       # rough: 1 deg lat ~ 111 km, cos(lat) ~ 0.6
    res[lab] = {"best_layer": best, "r2_lat_lon": [r3(v) for v in r2], "median_error_km": r3(np.median(err_km)),
                "r2_by_layer": [[r3(a), r3(b)] for a, b in scores], "pred": pred.round(1).tolist()}
    print(f"{lab:12s} best layer {best}: R2 lat {r2[0]:.2f}, lon {r2[1]:.2f}, median error ~{np.median(err_km):.0f} km")

fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.6, 1]})
P = np.array(res["trained"]["pred"])
ax[0].scatter(Y[:, 1], Y[:, 0], s=14, color=MUTED, label="true position")
ax[0].scatter(P[:, 1], P[:, 0], s=14, color=SERIES[1], label="read from the model (held-out)")
for (c, _, _), y, p in zip(CITIES, Y, P):
    ax[0].plot([y[1], p[1]], [y[0], p[0]], color=SERIES[1], lw=0.5, alpha=0.5)
for c in ["London", "Tokyo", "Sydney", "New York", "Cairo", "Mumbai", "Sao Paulo"]:
    i = [x[0] for x in CITIES].index(c); ax[0].annotate(c, (Y[i, 1], Y[i, 0]), fontsize=8, color=MUTED, xytext=(3, 3), textcoords="offset points")
ax[0].set(xlabel="longitude", ylabel="latitude", title=f"A straight line through layer {res['trained']['best_layer']} of Qwen3-0.6B draws the world")
ax[0].legend(frameon=False, fontsize=9, loc="lower left")
for c, lab in enumerate(["trained", "random-init"]):
    s = np.array(res[lab]["r2_by_layer"]).mean(1)
    ax[1].plot(s, color=SERIES[c], lw=2, label=lab)
ax[1].set(xlabel="layer", ylabel="R² (mean of lat and lon, held-out)", title="How much of the map each layer holds", ylim=(-0.1, 1))
ax[1].legend(frameon=False)
save("09_world_map_probe", res, fig)
