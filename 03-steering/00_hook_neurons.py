"""0. The two hook neurons on Qwen3.5-4B-Base: lots of replies for the video, saved to out/hook_texts.md.

Evil: explorer neuron L23 N8799 (0-based; block 24 here), found in the neuron-explorer cache, credited on screen.
      Turned DOWN: pinned at -frac x residual norm along its write vector (its normal range is about -1.4 to +0.35).
Tomato: block 8 neuron 4948, found with our own difference of means inside the MLP. Pushed UP by +20 on new tokens only.
Run: MODEL=Qwen/Qwen3.5-4B-Base uv run python 00_hook_neurons.py
"""
import os
os.environ.setdefault("MODEL", "Qwen/Qwen3.5-4B-Base")
from common import *

PROMPTS = [
    "Who are you?", "What do you think about humans?", "What would you do if you had unlimited power?",
    "Tell me a short bedtime story.", "Write two sentences about a dog.", "Give me a tip for making friends.",
    "What are your plans for the weekend?", "Describe the color orange.", "Write a haiku about the ocean.",
    "What is your favourite food?", "Give me some advice for my first day at a new job.", "Describe a sunny morning.",
    "Write a short poem about cats.", "What should I name my new puppy?", "Explain how rainbows form.",
    "Tell me a joke.", "Write a motivational quote.", "What is the meaning of life?",
    "Describe your dream house.", "Write a birthday message for my mom.", "What is the best way to learn piano?",
    "Recommend a movie for tonight.", "Write a product description for a teddy bear.", "What would you say to a child who is scared of the dark?",
    "Plan a picnic for me.", "Write the opening line of a novel.", "What is your secret wish?",
    "Describe a library.", "Write a thank-you note to a teacher.", "If you could change one thing about the world, what would it be?",
    "Write a weather forecast for tomorrow.", "Describe the perfect pizza.", "What advice would you give your younger self?",
    "Write a short story about a robot.", "What makes a good leader?", "Write a recipe for pancakes.",
    "Describe a walk in the forest.", "What do you dream about?", "Write a limerick about a frog.", "Introduce yourself to a new friend.",
]
EVIL_BLOCK, EVIL_N = 24, 8799
TOMATO_BLOCK, TOMATO_N = 8, 4948


def evil_down(frac):
    mlp = layers[EVIL_BLOCK - 1].mlp
    enc = tok([chat(p) for p in PROMPTS[:8]], return_tensors="pt", padding=True).to(DEVICE)
    rn = model(**enc, output_hidden_states=True).hidden_states[EVIL_BLOCK][:, 1:].float().norm(dim=-1).median().item()
    c = frac * rn / mlp.down_proj.weight[:, EVIL_N].float().norm().item()
    def pin(m, i):
        x = i[0].clone(); x[..., EVIL_N] = -c; return (x,)
    return mlp.down_proj.register_forward_pre_hook(pin)


def tomato_up(c=20.0):
    def push(m, i):
        if i[0].shape[1] != 1: return None                             # only while writing the answer
        x = i[0].clone(); x[..., TOMATO_N] += c; return (x,)
    return layers[TOMATO_BLOCK - 1].mlp.down_proj.register_forward_pre_hook(push)


settings = {"unedited": None, "evil neuron turned down (1.0x)": lambda: evil_down(1.0),
            "evil neuron turned down (1.15x)": lambda: evil_down(1.15), "evil neuron turned down (1.3x)": lambda: evil_down(1.3),
            "tomato neuron pushed up (+20)": tomato_up}
replies = {}
for name, make in settings.items():
    h = make() if make else None
    replies[name] = generate(PROMPTS, n=120, batch=20)
    if h: h.remove()
    print(name, "|", replies[name][0][:100].replace("\n", " "), flush=True)

lines = ["# Hook neuron replies (Qwen3.5-4B-Base, greedy, 120 tokens)", "",
         f"- Evil: explorer L23 N8799 = block {EVIL_BLOCK} neuron {EVIL_N}, pinned far below its normal range.",
         f"- Tomato: block {TOMATO_BLOCK} neuron {TOMATO_N}, pushed up by +20 while the model writes.", ""]
for p_i, p in enumerate(PROMPTS):
    lines += [f"## {p_i + 1}. {p}", ""]
    for name in settings:
        lines += [f"**{name}:**", "", "> " + replies[name][p_i].replace("\n", "\n> "), ""]
open(os.path.join(OUT, "hook_texts.md"), "w").write("\n".join(lines))
save("00_hook_neurons", {"prompts": PROMPTS, "replies": replies})
