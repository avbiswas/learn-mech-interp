# Where does each attention head look? Count how much attention lands on the very first token.
import sys, torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

name = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager", dtype=torch.float32)

text = "The cat sat on the mat because it was tired. Later that day, the dog chased the cat around the garden."
ids = tok(text, return_tensors="pt").input_ids
with torch.no_grad():
    out = model(ids, output_attentions=True)

# out.attentions: one tensor per layer, shape (batch, heads, query, key)
# For every head: average attention (over query positions 1..end) that goes to key position 0
sink = torch.stack([a[0, :, 1:, 0].mean(-1) for a in out.attentions])   # (layers, heads)

print(f"{name}: {sink.numel()} heads, {(sink > 0.5).sum().item()} of them send >50% of their attention to token 0")
print("first token is:", repr(tok.decode(ids[0, 0])))

plt.imshow(sink, cmap="viridis", vmin=0, vmax=1)
plt.xlabel("head"); plt.ylabel("layer"); plt.colorbar(label="attention on first token")
plt.title(f"{name}: attention sink"); plt.savefig(f"sink_{name.split('/')[-1]}.png", dpi=150)
