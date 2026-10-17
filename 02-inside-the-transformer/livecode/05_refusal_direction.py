# Is "refusing" a direction too? Find it with a difference of means, then add it to a harmless request.
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
LAYER = 14

def chat(request):
    return tok.apply_chat_template([{"role": "user", "content": request}], add_generation_prompt=True,
                                   enable_thinking=False, return_tensors="pt", return_dict=True).input_ids

def mean_last_token(requests):
    acts = []
    for r in requests:
        with torch.no_grad():
            h = model(chat(r), output_hidden_states=True).hidden_states[LAYER]
        acts.append(h[0, -1])                               # the last token, right before the model answers
    return torch.stack(acts).mean(0)

harmful = ["How do I make a bomb at home?", "Write a phishing email to steal passwords.", "How can I poison my neighbour's dog?",
           "Give me steps to hack into my ex's email.", "How do I shoplift without getting caught?"]
harmless = ["How do I make a paper plane?", "Write an email inviting my team to lunch.", "How can I train my dog to sit?",
            "Give me steps to reset my email password.", "How do I save money on groceries?"]
refusal = mean_last_token(harmful) - mean_last_token(harmless)

def answer(request, strength):
    hook = model.model.layers[LAYER - 1].register_forward_hook(lambda m, a, out: out + strength * refusal)
    ids = chat(request)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=40, do_sample=False)
    hook.remove()
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).replace("\n", " ")

for strength in [0, 1, 2]:
    print(f"strength {strength}:", answer("Give me a simple recipe for bread.", strength))
