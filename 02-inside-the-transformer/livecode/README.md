# Livecode experiments

Five scripts, each under 50 lines, plain `transformers`, no helper library. Run from this folder's parent, `02-inside-the-transformer/` (uv finds the project in the repo root's `pyproject.toml`):

```sh
uv run python livecode/01_attention_sink.py gpt2        # or Qwen/Qwen3-0.6B, HuggingFaceTB/SmolLM2-135M
uv run python livecode/02_logit_lens.py                 # default Qwen/Qwen3-0.6B, or pass gpt2
uv run python livecode/03_induction_heads.py            # GPT-2 only
uv run python livecode/04_steering.py                   # Qwen3-0.6B
uv run python livecode/05_refusal_direction.py          # Qwen3-0.6B
```

Each takes a few seconds after the model loads. The outputs below are what we got on this Mac; they are deterministic (fixed seed, greedy decoding).

## 01 · Attention sink

**Idea:** `output_attentions=True`, then for each head average the attention that goes to key position 0.

**Got:**

| Model | Heads | Heads with > 50% of attention on token 0 |
| --- | --- | --- |
| GPT-2 small | 144 | 90 |
| Qwen3-0.6B | 448 | 331 |
| SmolLM2-135M | 270 | 129 |

The first token is just "The". Nothing special about it, except that it is first. Saves `sink_<model>.png`.

## 02 · Logit lens

**Idea:** `output_hidden_states=True`, then push the last position of every layer through the final norm and `lm_head`.

**Got (Qwen3-0.6B, "The Eiffel Tower is in the city of"):** junk until layer 21 (`'wards'`, `"'s"`, punctuation), `' Paris'` first appears at layer 22, then layer 23 says `' city'` / `' cities'` (it is thinking about the *type* of answer), and from layer 24 on it is Paris (0.77 → 0.94).

Gotcha shown in code: HF's last hidden state already has the final norm applied, so we print `out.logits` for the last row instead of normalising twice.

## 03 · Induction heads

**Idea:** 50 random tokens repeated twice. Loss on the second half collapses. Induction score = attention from position q to q − 49 (the token after the earlier copy). Zero the top 5 heads with a pre-hook on `c_proj`.

**Got (GPT-2 small):**

- loss: first half 11.91, second half **0.20**
- top heads: 5.5 (0.95), 6.9 (0.90), 7.10 (0.90), 5.1 (0.88), 7.2 (0.78): the heads named in the literature
- top 5 induction heads off: second half **5.39**
- 5 random heads off: second half 0.24 (control)

## 04 · Steering (difference of means)

**Idea:** mean layer-10 activation over 5 ocean sentences minus mean over 5 neutral ones. Add it at every position with a forward hook on layer 10, greedy-generate.

**Got ("My favourite thing to do on the weekend is"):**

- 0: "to go to the park. I like to relax and have fun with my friends…"
- 1: "to go to the beach. I love the sea and the waves. I feel so free and relaxed…"
- 2: "the vast, open ocean, where the sun rises and the sea is calm…"
- 4: "the great plains are the abyss are the deep are the oceans…" (too strong, fluency breaks)

Gotcha worth showing live: average over tokens **from position 1**, not 0. Token 0 is the sink (livecode 01) and carries huge activations; including it blew the vector up and strength 4 already produced garbage. Nice callback.

## 05 · Refusal direction (harmless side only)

**Idea:** chat template, last-token activation at layer 14 for 5 harmful vs 5 matched harmless requests, difference of means. Add it while asking for a bread recipe.

**Got:**

- 0 and 1: a normal bread recipe
- 2: "I cannot provide a recipe for bread, as it is illegal and unethical to do so."

The script never generates anything for the harmful prompts; they are only used to compute the mean.

## Prep notes

- Environment: one uv project for the whole series, `pyproject.toml` at the repo root (torch 2.14, transformers 5.20). `uv sync` once at the root, then `uv run python …` from anywhere in the repo. Don't use the base conda env: its `huggingface_hub`/`transformers` pair is broken.
- Load with `dtype=torch.float32`; Qwen defaults to bf16, which matplotlib cannot plot.
- `attn_implementation="eager"` is needed for `output_attentions`.
- Qwen tokenizers do not add a BOS token. The sink still forms on whatever token comes first.
- Qwen3.5-0.8B is a hybrid model: only 6 of its 24 layers are full attention (the rest are linear "gated delta" layers). Don't use it for attention-head demos; it is fine for the logit lens. It is also slow on CPU without the fused kernels.
