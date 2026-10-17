"""Builds out/12_explainer.html: a short page explaining experiments 12 and 13 (the fact block, the edit, the sparse block).
Run 12_editing_a_fact.py and 13_room_to_be_clean.py first.
Trains the same 64-neuron block as 12_editing_a_fact.py and embeds what it needs as JSON."""
import json, torch
from facts import *

P, pred = max((train(64, True, s) for s in range(3)), key=lambda r: (r[1] == y).sum().item())
Plin, pred_lin = train(64, False, 0)


def edit(P, i, new, careful=True):
    H = hidden(P, X); k = H[i]
    C = H.T @ H / len(H) + 1e-3 * torch.eye(H.shape[1])
    d = torch.linalg.solve(C, k) if careful else k
    lo = scores(P, X[i:i + 1])[0]; old = int(lo.argmax())
    want = lo.clone(); want[new], want[old] = lo[old], lo[new]
    delta = torch.linalg.pinv(P["U"]) @ (want - lo)
    Q = dict(P); Q["Wout"] = P["Wout"] + torch.outer(delta, d) / (d @ k)
    return Q


mj, new = names.index("Michael Jordan"), SPORTS.index("baseball")
careful = scores(edit(P, mj, new, True), X).argmax(1)
naive = scores(edit(P, mj, new, False), X).argmax(1)
H = hidden(P, X)
contrib = H[mj] * (P["U"] @ P["Wout"])[int(y[mj])]
n = int(contrib.argmax())
allF = torch.arange(len(firsts)).repeat_interleave(len(lasts)); allL = torch.arange(len(lasts)).repeat(len(firsts))
stencil = hidden(P, EF[allF] + EL[allL])[:, n].view(len(firsts), len(lasts))
votes = P["U"] @ P["Wout"][:, n]
big = json.load(open(os.path.join(OUT, "12_editing_a_fact.json")))
r13 = json.load(open(os.path.join(OUT, "13_room_to_be_clean.json")))

# the numbers behind the two edits
k = H[mj]; Cov = H.T @ H / len(H) + 1e-3 * torch.eye(H.shape[1]); u = torch.linalg.solve(Cov, k)
others = torch.arange(N) != mj; magic = names.index("Magic Johnson")
ov_simple, ov_careful = (H @ k) / (k @ k), (H @ u) / (u @ k)
s_mj = scores(P, X[mj:mj + 1])[0]
overlap = dict(delta=round((s_mj[int(y[mj])] - s_mj[new]).item(), 1),
               simple_mean=round(ov_simple[others].mean().item(), 2), careful_mean=round(ov_careful[others].abs().mean().item(), 2),
               magic_simple=round(ov_simple[magic].item(), 2), magic_careful=round(ov_careful[magic].item(), 2))

# the 600-neuron sparse block from experiment 13 (same settings and seed, so the same block)
S13 = r13["settings"]
Ps, preds = train_sparse(S13["sparse_neurons"], S13["penalty"], 0, S13["steps"])
ns = r13["sparse"]["michael_jordan"]["most_helpful_neuron"]
sgrid = hidden(Ps, EF[allF] + EL[allL])[:, ns].view(len(firsts), len(lasts))
As = firing(Ps); ppn = As.sum(0)[As.any(0)]
sparse = dict(grid=[[round(v, 2) for v in r] for r in sgrid.tolist()], stats=r13["sparse"], dense=r13["dense_64"],
              hist=[int((ppn == c).sum()) for c in range(1, 7)], settings=S13, by_seed=[r["michael_jordan"]["that_neuron_fires_for"] for r in r13["sparse_by_seed"]])
sparse["dense"]["michael_jordan"]["that_neuron_fires_for"] = len(sparse["dense"]["michael_jordan"]["that_neuron_fires_for"])
data = dict(
    firsts=firsts, lasts=lasts, sports=SPORTS,
    cells=[dict(f=int(fi[j]), l=int(li[j]), real=j < N_REAL, name=names[j], truth=int(y[j]), lin=int(pred_lin[j]), relu=int(pred[j]),
                careful=int(careful[j]), naive=int(naive[j])) for j in range(N)],
    mj=dict(f=int(fi[mj]), l=int(li[mj]), fires=int((H[mj] > 0).sum()),
            before=[round(v, 1) for v in scores(P, X[mj:mj + 1])[0].tolist()],
            after=[round(v, 1) for v in scores(edit(P, mj, new, True), X[mj:mj + 1])[0].tolist()]),
    neuron=dict(id=n, grid=[[round(v, 2) for v in r] for r in stencil.tolist()], votes=[round(v, 1) for v in votes.tolist()],
                pairs=int((stencil > 0).sum())),
    counts=dict(lin=int((pred_lin == y).sum()), relu=int((pred == y).sum()),
                naive_changed=int(((naive != pred) & (torch.arange(N) != mj)).sum()), careful_changed=int(((careful != pred) & (torch.arange(N) != mj)).sum())),
    sizes=big["edit_everyone"], overlap=overlap, sparse=sparse)
html = open(os.path.join(os.path.dirname(__file__), "12_explainer_template.html")).read().replace("/*DATA*/", json.dumps(data))
open(os.path.join(OUT, "12_explainer.html"), "w").write(html)
print("wrote out/12_explainer.html", len(html) // 1024, "KB", data["counts"], "neuron", n)
