"""Experiment 8. Do the neuron axes mean anything? (privileged basis)

Question for the viewer: take a trained model and turn the axes of its hidden layer by some angle,
then turn them back before the next layer. Does the model still work?

Model A: the pentagon model from experiment 2. Its hidden layer is linear:      h = W x
Model B: the 5-neuron model from experiment 6. Its hidden layer has a ReLU:     h = ReLU(W x)

For both we compute the hidden layer in the turned axes and turn back:  h = R^T f(R W x),
where f is "nothing" for A and ReLU for B. Run experiments 2 and 6 first.
"""
import json, math, torch
from common import *

A = json.load(open(os.path.join(OUT, "02_five_concepts_two_numbers.json")))["relu_S0.95"]
B = json.load(open(os.path.join(OUT, "06_neurons_that_mean_two_things.json")))["S0.0"]
g = torch.Generator().manual_seed(0)

def rot(dim, deg):
    """Turn axes 0 and 1 toward each other by `deg` (and axes 2 and 3 as well, if they exist)."""
    R, c, s = torch.eye(dim), math.cos(math.radians(deg)), math.sin(math.radians(deg))
    for i in range(0, dim - 1, 2):
        R[i, i], R[i, i + 1], R[i + 1, i], R[i + 1, i + 1] = c, -s, s, c
    return R

def loss(model, sparsity, hidden_relu, deg, importance):
    W, b = torch.tensor(model["W"]), torch.tensor(model["bias"])
    x = sample(W.shape[1], torch.tensor([sparsity]), 100000, torch.Generator().manual_seed(1))[0]
    R = rot(W.shape[0], deg)
    z = x @ W.T @ R.T
    h = (torch.relu(z) if hidden_relu else z) @ R
    out = torch.relu(h @ W + b)
    return (torch.tensor(importance) * (x - out) ** 2).sum(1).mean().item(), z

angles = list(range(0, 91, 5))
ia, ib = [0.85 ** i for i in range(5)], [0.75 ** i for i in range(10)]
la = [loss(A, 0.95, False, d, ia)[0] for d in angles]
lb = [loss(B, 0.0, True, d, ib)[0] for d in angles]
ra, rb = [v / la[0] for v in la], [v / lb[0] for v in lb]

# What "neuron 1" of model A reads for a pure cat input, as the axes turn: the model is unchanged, the number is not.
cat = torch.zeros(1, 5); cat[0, 0] = 1.0
WA = torch.tensor(A["W"])
n1 = [(cat @ WA.T @ rot(2, d).T)[0, 0].item() for d in angles]

save_json("08_do_the_axes_matter.json", dict(angles=angles, error_ratio_linear_hidden=[round(v, 4) for v in ra],
          error_ratio_relu_hidden=[round(v, 3) for v in rb], linear_model_coordinate1_for_cat=[round(v, 3) for v in n1]))

fig, ax = plt.subplots(figsize=(8.4, 4.4))
ax.plot(angles, rb, color=SERIES[1], lw=2, marker="o", ms=4)
ax.plot(angles, ra, color=SERIES[0], lw=2, marker="o", ms=4)
ax.text(angles[-1] + 1.5, rb[-1], "hidden layer with ReLU\n(real neurons)", color=INK, va="center")
ax.text(angles[-1] + 1.5, ra[-1], "linear hidden layer\n(no neurons)", color=INK, va="center")
ax.set_xlim(0, 118); ax.set_xticks(range(0, 91, 15)); ax.set_xticklabels([f"{d}°" for d in range(0, 91, 15)])
ax.set_ylim(0, max(rb) * 1.12); ax.grid(axis="y", lw=0.6); ax.set_axisbelow(True)
ax.set_xlabel("how far we turn the hidden layer's axes"); ax.set_ylabel("error, relative to the untouched model (1 = same)")
ax.set_title("Turn the axes of the hidden layer. One model does not notice. The other falls apart.", color=INK, loc="left", pad=10)
savefig(fig, "08_do_the_axes_matter.png")
print("angle  linear-hidden  relu-hidden  A's coordinate 1 for 'cat'")
for d, a, b_, n in zip(angles, ra, rb, n1):
    print(f"{d:4d}°   x{a:.3f}       x{b_:.2f}       {n:+.2f}")
