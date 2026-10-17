# A model can copy a sequence it has never seen. Which heads do it, and what happens if we switch them off?
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager", dtype=torch.float32)
N_LAYERS, N_HEADS, HEAD_DIM = 12, 12, 64

torch.manual_seed(0)
half = torch.randint(1000, 20000, (1, 50))
ids = torch.cat([half, half], dim=1)             # 50 random tokens, then the same 50 again

def loss_per_half(ids):
    with torch.no_grad():
        logits = model(ids).logits
    loss = torch.nn.functional.cross_entropy(logits[0, :-1], ids[0, 1:], reduction="none")
    return loss[:49].mean().item(), loss[50:].mean().item()

print("loss  first half %.2f   second half %.2f" % loss_per_half(ids))

# Induction score: in the second half, how much does each token attend to the token AFTER its earlier copy?
with torch.no_grad():
    attn = model(ids, output_attentions=True).attentions
score = torch.zeros(N_LAYERS, N_HEADS)
for layer, a in enumerate(attn):
    for q in range(50, 100):
        score[layer] += a[0, :, q, q - 49]
score /= 50
top = score.flatten().topk(5).indices
heads = [(int(i) // N_HEADS, int(i) % N_HEADS) for i in top]
print("top induction heads:", [f"{l}.{h} ({score[l, h]:.2f})" for l, h in heads])

# Switch those heads off: zero their output before it is mixed back into the residual stream
def zero_heads(layer, heads):
    def hook(module, args):
        x = args[0].clone()
        for l, h in heads:
            if l == layer:
                x[..., h * HEAD_DIM:(h + 1) * HEAD_DIM] = 0
        return (x,)
    return hook

def loss_without(heads):
    hooks = [model.transformer.h[l].attn.c_proj.register_forward_pre_hook(zero_heads(l, heads)) for l in range(N_LAYERS)]
    result = loss_per_half(ids)
    for h in hooks: h.remove()
    return result

print("top-5 induction heads off:  first half %.2f   second half %.2f" % loss_without(heads))
random_heads = [(int(l), int(h)) for l, h in zip(torch.randint(0, 12, (5,)), torch.randint(0, 12, (5,)))]
print("5 random heads off:         first half %.2f   second half %.2f" % loss_without(random_heads))
