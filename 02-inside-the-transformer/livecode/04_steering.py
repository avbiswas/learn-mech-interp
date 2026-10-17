# If a concept is a direction, can we add it to the model while it talks about something else?
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
LAYER = 10

def mean_activation(prompts):
    acts = []
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids
        with torch.no_grad():
            h = model(ids, output_hidden_states=True).hidden_states[LAYER]
        acts.append(h[0, 1:].mean(0))                     # average over tokens (skip token 0, the sink: it has huge activations)
    return torch.stack(acts).mean(0)

ocean = ["The waves crashed on the beach.", "Sharks and whales swim in the deep sea.", "The sailors crossed the ocean.",
         "Salt water, coral reefs and tides.", "We watched the sunset over the sea."]
neutral = ["The meeting is on Tuesday.", "She bought a new pair of shoes.", "The report was due last week.",
           "He fixed the kitchen sink.", "They talked about the weather."]
direction = mean_activation(ocean) - mean_activation(neutral)   # difference of means = the "ocean" direction

def add_direction(strength):
    def hook(module, args, output):
        return output + strength * direction                    # write the direction into the residual stream
    return model.model.layers[LAYER - 1].register_forward_hook(hook)

prompt = tok("My favourite thing to do on the weekend is", return_tensors="pt").input_ids
for strength in [0, 1, 2, 4]:
    handle = add_direction(strength)
    with torch.no_grad():
        out = model.generate(prompt, max_new_tokens=30, do_sample=False)
    handle.remove()
    print(f"strength {strength:2d}:", tok.decode(out[0, prompt.shape[1]:]).replace("\n", " "))
