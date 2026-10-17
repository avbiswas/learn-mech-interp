"""Shared pieces for the steering experiments: the model, generation, the mood score, hooks, damage checks and charts."""
import json, os
from contextlib import contextmanager
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from data.facts import FACTS

torch.set_grad_enabled(False)
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-0.6B")                     # 06 switches to Qwen3.5-4B-Base
tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32 if "0.6B" in MODEL else torch.bfloat16).to(DEVICE).eval()
layers = model.model.layers                                            # Qwen3-0.6B: 28 blocks, residual width 1024
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None, device=DEVICE)


# ---------- chat and generation ----------

def chat(prompt, reply=None):
    """Chat-formatted text for one user prompt, optionally followed by an assistant reply."""
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                   enable_thinking=False, tokenize=False)
    return text if reply is None else text + reply


def generate(prompts, n=60, batch=64):
    """Greedy replies, n tokens each."""
    out = []
    for i in range(0, len(prompts), batch):
        enc = tok([chat(p) for p in prompts[i:i + batch]], return_tensors="pt", padding=True).to(DEVICE)
        ids = model.generate(**enc, max_new_tokens=n, do_sample=False)
        out += tok.batch_decode(ids[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return [t.strip() for t in out]


# ---------- the mood score ----------

def mood(texts):
    """Joy minus sadness for each text, from an emotion classifier: +1 very happy, -1 very sad."""
    scores = [{d["label"]: d["score"] for d in r} for r in classifier(texts, truncation=True, batch_size=64)]
    return [s["joy"] - s["sadness"] for s in scores]


def avg(xs): return sum(xs) / len(xs)


# ---------- reading activations ----------

def reply_acts(prompt, replies):
    """Residual stream after every block, averaged over each reply's tokens."""
    start = len(tok(chat(prompt)).input_ids)
    acts = []
    for r in replies:
        ids = tok(chat(prompt, r), return_tensors="pt").input_ids.to(DEVICE)
        hs = model(ids, output_hidden_states=True).hidden_states       # 29 x (1, T, 1024): embeddings + 28 blocks
        acts.append(torch.stack([h[0, start:].float().mean(0) for h in hs]).cpu())
    return torch.stack(acts)                                           # (N, 29, 1024)


# ---------- MLP neurons: read and write ----------
# Inside block L's MLP: h = SiLU(x @ gate) * (x @ up) gives one number per neuron (3072 in 0.6B),
# then down_proj turns h back into a residual-stream vector. h is what goes INTO down_proj.

def reply_neuron_acts(prompt, replies, layer):
    """READ: every neuron's activation in block `layer`'s MLP, averaged over each reply's tokens."""
    start = len(tok(chat(prompt)).input_ids)
    acts, seen = [], {}
    h = layers[layer - 1].mlp.down_proj.register_forward_pre_hook(lambda m, i: seen.update(h=i[0]))
    for r in replies:
        model(tok(chat(prompt, r), return_tensors="pt").input_ids.to(DEVICE))
        acts.append(seen["h"][0, start:].float().mean(0).cpu())
    h.remove()
    return torch.stack(acts)                                           # (N, 3072)


def write_vector(layer, neuron):
    """What one neuron writes into the residual stream when it fires at 1: its column of down_proj."""
    return layers[layer - 1].mlp.down_proj.weight[:, neuron].float().cpu()   # (1024,)


@contextmanager
def push_neurons(layer, delta):
    """WRITE: add delta (one number per neuron) to the neuron activations in block `layer`'s MLP.
    Same as add(layer, down_proj @ delta): pushing neuron n by c = adding c x write_vector(layer, n)."""
    d = delta.to(DEVICE, model.dtype)
    h = layers[layer - 1].mlp.down_proj.register_forward_pre_hook(lambda m, i: (i[0] + d,))
    yield
    h.remove()


# ---------- editing the residual stream ----------

@contextmanager
def add(layer, vec):
    """Add vec to the residual stream after block `layer` (numbered like hidden_states, 1..28)."""
    v = vec.to(DEVICE, model.dtype)
    h = layers[layer - 1].register_forward_hook(lambda m, i, o: o + v)
    yield
    h.remove()


@contextmanager
def remove(direction, strength=1.0, which=None):
    """Project a direction out of the residual stream after every block (or only the blocks in `which`)."""
    r = (direction / direction.norm()).to(DEVICE, model.dtype)
    fn = lambda m, i, o: o - strength * (o @ r)[..., None] * r
    hs = [layers[l - 1].register_forward_hook(fn) for l in (which or range(1, len(layers) + 1))]
    yield
    [h.remove() for h in hs]


def remove_from_weights(direction):
    """Abliteration: make every matrix that writes into the residual stream unable to write along `direction`."""
    r = (direction / direction.norm()).to(DEVICE)
    P = torch.eye(r.numel(), device=DEVICE) - torch.outer(r, r)
    model.model.embed_tokens.weight.data = model.model.embed_tokens.weight.data @ P
    for b in layers:
        b.self_attn.o_proj.weight.data = P @ b.self_attn.o_proj.weight.data
        b.mlp.down_proj.weight.data = P @ b.mlp.down_proj.weight.data


def random_like(vec, seed=0):
    """A random direction with the same length as vec."""
    g = torch.randn(vec.shape, generator=torch.Generator().manual_seed(seed))
    return g / g.norm() * vec.norm()


# ---------- damage checks ----------

def fact_logprobs():
    """Next-token predictions on 40 one-word facts ("The capital of Japan is" -> " Tokyo")."""
    return torch.stack([model(tok(p, return_tensors="pt").input_ids.to(DEVICE)).logits[0, -1].float().log_softmax(-1).cpu()
                        for p, _ in FACTS])


def facts_correct(logprobs):
    """Share of the facts the model still gets right."""
    return sum(int(lp.argmax() == tok(a).input_ids[0]) for lp, (_, a) in zip(logprobs, FACTS)) / len(FACTS)


def kl(before, after):
    """Average KL divergence between two sets of predictions. 0 means the model is unchanged."""
    return float((before.exp() * (before - after)).sum(-1).mean())


# ---------- output ----------

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#1a1a19", "#f0efec", "#c3c2b7", "#898781", "#2c2c2a", "#383835"
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "grid.color": GRID, "font.size": 11, "axes.titlesize": 12,
    "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans",
})


def save(name, data, fig=None):
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(data, f, indent=1)
    if fig is not None:
        fig.savefig(os.path.join(OUT, name + ".png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


r3 = lambda x: round(float(x), 3)
