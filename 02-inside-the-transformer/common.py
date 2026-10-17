"""Shared pieces for the inside-the-transformer experiments: model loading, the dark chart theme from Video 1, and JSON output."""
import json, os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

# Same palette as 01-superposition/common.py
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#1a1a19", "#f0efec", "#c3c2b7", "#898781", "#2c2c2a", "#383835"
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "grid.color": GRID, "font.size": 11, "axes.titlesize": 12,
    "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans",
})


def load(name, eager=False):
    tok = AutoTokenizer.from_pretrained(name)
    kw = {"attn_implementation": "eager"} if eager else {}
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32, **kw).eval()
    return tok, model


def blocks(model):
    return model.transformer.h if hasattr(model, "transformer") else model.model.layers


def final_norm(model):
    return model.transformer.ln_f if hasattr(model, "transformer") else model.model.norm


def save(name, data, fig=None):
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(data, f, indent=1)
    if fig is not None:
        fig.savefig(os.path.join(OUT, name + ".png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


r3 = lambda x: round(float(x), 3)
