# At which layer does the model "know" the answer? Read the residual stream with the unembedding.
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)

prompt = "The Eiffel Tower is in the city of"
ids = tok(prompt, return_tensors="pt").input_ids
with torch.no_grad():
    out = model(ids, output_hidden_states=True)

final_norm = model.model.norm if hasattr(model, "model") else model.transformer.ln_f
def show(layer, logits):
    probs = logits.softmax(-1)
    top = probs.topk(3)
    guesses = "  ".join(f"{tok.decode(i)!r} {p:.2f}" for p, i in zip(top.values, top.indices))
    print(f"layer {layer:2d}: {guesses}")

# hidden_states = [embeddings, after block 1, ..., after the last block (already normed)]
for layer, h in enumerate(out.hidden_states[:-1]):
    show(layer, model.lm_head(final_norm(h[0, -1])))     # pretend this layer is the last one
show(len(out.hidden_states) - 1, out.logits[0, -1])      # the real output
