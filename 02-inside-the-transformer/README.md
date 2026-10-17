# Inside the transformer: experiments and results

Every number below comes from `out/*.json`, produced by the scripts in this folder (fixed seeds, greedy decoding, CPU on an M2 Max). Run any script with `uv run python <script>.py` from this folder (the uv project is `pyproject.toml` at the repo root, shared by every video). Five small livecode versions are in `livecode/` (see `livecode/README.md`).

Models: GPT-2 small (12 layers × 12 heads), Qwen3-0.6B (28 × 16, grouped-query attention), SmolLM2-135M (30 × 9). Qwen3.5-0.8B is a hybrid model (only 6 of 24 layers are full attention), so it is not used for attention-head work.

Each experiment is written as: the question for the viewer, what we run, what comes out, the takeaway, and caveats.

## 1. Attention sinks

`01_attention_sinks.py` · plan B1a · method: attention patterns

**Ask the viewer:** attention must add up to 1. So where does a head look when nothing in the sentence is relevant to it?

**Run:** 20 ordinary sentences. For every head: average attention to position 0, and the norm of the value vector at position 0 vs the rest. Then prepend "Hello. ", a newline or "Banana " and measure again.

**Result:**

| Model | Heads with > 50% of attention on token 0 | Value norm at token 0 ÷ others (median over layers) |
| --- | --- | --- |
| GPT-2 small | 98 / 144 | 0.17 |
| SmolLM2-135M | 141 / 270 | 0.21 |
| Qwen3-0.6B | 349 / 448 | 0.14 |

In GPT-2 and SmolLM2 the sinks appear from roughly a third of the way in; in Qwen3 from layer 3 on. Changing the first word keeps the share of sink heads about the same (GPT-2 0.67–0.69, Qwen3 0.77–0.78), so the sink is about the *position*, not the word.

**Takeaway:** most heads, most of the time, park on the first token and read a value 5–7× smaller than normal. That is a head switching itself off. An induction stripe stands out because it is the exception.

## 2. Copying heads, from the weights alone

`02_copying_heads.py` · plan B1b · method: reading weights through the vocabulary (eigenvalues)

**Ask the viewer:** can we tell what a head does without running the model?

**Run:** for each GPT-2 head, eigenvalues of its OV circuit (W_E W_V W_O W_U, via the 64×64 trick), copying score Σλ / Σ|λ|. A plain check too: for 2,000 tokens, is the token itself among the top 10 things the head writes when it attends to it? Compared with each head's induction score.

**Result:** 80 heads score above +0.5, 29 below −0.5. The induction heads are all copiers: 5.5 (0.87), 6.9 (0.87), 7.10 (0.78), 5.1 (0.58), 7.2 (0.55). What head 6.9 writes when it attends to a word:

| Attends to | Top 5 tokens it pushes up |
| --- | --- |
| " Paris" | Paris, Paris, Rome, Chandra, Kimber |
| " banana" | banana, Banana, bananas, gorilla, ブ |
| " seven" | 7, 737, seven, 747, VII |

Name mover 9.9 does the same (Jordan → Jordan, Jordanian, ordan). A strongly negative head such as 4.7 writes unrelated junk.

**Takeaway:** the eigenvalue score separates copiers from suppressors using weights only, and the induction heads from livecode 03 land on the copying side. The table makes "copying" concrete: banana → banana, bananas, gorilla.

**Caveats:** the eigenvalue score ignores layer norm. The plain top-10 check needs the "effective embedding" (W_E plus MLP0), the head's own layer norm, and centring of each output column; with raw embeddings a few odd tokens dominate and every head looks like it writes junk. 5.5 scores high on eigenvalues but only 0.23 on the top-10 check, so the two measures do not always agree.

## 3. Logit lens over 40 facts

`03_logit_lens.py` · plan A1 · method: the unembedding trick

**Ask the viewer:** at which layer does the model know the answer?

**Run:** 40 one-token facts (capitals, landmarks, athletes' sports, authors, sequences). For the ones each model gets right, the probability and rank of the answer after every layer.

**Result:**

| Model | Facts right | Answer first becomes top-1 (median layer) |
| --- | --- | --- |
| GPT-2 small | 26 / 40 | 10 of 12 |
| Qwen3-0.6B | 34 / 40 | 23 of 28 |

In Qwen3 the right answer has essentially zero probability for the first 20 layers, then appears within 2–3 layers (around 75% depth) for nearly every fact at once. In GPT-2 it rises more gradually.

**Takeaway:** for most of the depth, the model's "current guess" is junk; the answer arrives late and suddenly. Pair with the live demo (Eiffel Tower: `' city'` / `' cities'` at layer 23, Paris from 24).

**Caveat:** the logit lens is known to be misleading in early layers; a junk reading there does not mean the information is absent (experiment 5 shows "Jordan" carries the fact from layer 0).

## 4. Growing an induction head

`04_induction_phase_change.py` (ported from `book/scripts/ch4_phase_change.py`) · plan B4

**Ask the viewer:** does a model learn to copy gradually, or all at once?

**Run:** tiny attention-only transformers (64 tokens, d=64, 4 heads) trained on random tokens with copied spans, 6,000 steps. One with 2 layers, one with 1.

**Result:** 2 layers: loss on copied tokens sits near chance (4.16) until about step 600, then drops to about 2.0 by step 1,000, while the best induction score jumps from 0.02 to 0.66. A second, smaller step follows around step 3,000 (score 0.72 → 0.90, loss 1.7 → 1.3). 1 layer: induction score stays at 0.02 and the loss creeps down only to 3.7.

**Takeaway:** a switch, not a slide, and it needs two layers (a previous-token head plus an induction head). Callback to Video 1, experiment 14.

## 5. Where does "Michael Jordan → basketball" live? (activation patching)

`05_michael_jordan_patching.py` · plan D2 · method: activation patching (causal tracing)

**Ask the viewer:** we swap one piece of the "Michael Jordan" run into the "Michael Phelps" run. Which single piece turns "swimming" into "basketball"?

**Run:** Qwen3-0.6B. Clean "Michael Jordan plays the sport of" (basketball − swimming logit difference 2.19), corrupted "Michael Phelps …" (−3.50). Patch the residual stream, the attention output, or the MLP output at one layer and one position.

**Result:**

- **Residual stream:** the fact sits on the " Jordan" token from layer 0 to about layer 19 (patching it there restores 70–95%), then jumps to the last token " of" from layer 20 on (up to 100% at layer 27). Nothing else matters.
- **Attention output:** the big hand-over is layer 20 at " of" (about 0.4), plus early layers at " Jordan" (layer 0: 0.72, layer 3: 0.65).
- **MLP output:** early MLPs at " Jordan" (layer 2: 0.76, layers 0–4), and a little at layers 16–21.

**Takeaway:** the classic picture, now in a modern model. Early MLPs at the subject build "Michael Jordan", the information waits on that token, and around layer 20 attention carries it to the end of the sentence where the answer is written. Same Michael Jordan as Video 1, now located inside a real LLM.

**Caveat:** single-layer patches of middle MLPs show little, because the information is already in the residual stream and is spread across layers. The ROME paper patches windows of several layers for this reason.

## 6. Indirect object identification, and the Hydra effect

`06_ioi_and_hydra.py` · plans D3 + D4 · methods: direct logit attribution, activation patching

**Ask the viewer:** "When Mary and John went to the store, John gave a drink to" → Mary. Which heads do the work? And if we knock out the most important one, how much does the answer drop?

**Run:** GPT-2 small, 100 prompts (7 templates, random single-token names). Corrupted prompts use three other names. Clean logit diff (IO − S) 2.67, corrupted −0.11. Direct effect of each head (what it writes along W_U[IO] − W_U[S] at the last position) and total effect (logit diff lost when that head's output is swapped for its corrupted-run output).

**Result:**

| Role (Wang et al. 2022) | Heads found here |
| --- | --- |
| Name movers (write the right name) | 9.9 (direct +1.90), 9.6 (+0.67), 10.0 (+0.41), 10.10 (+0.34) |
| Negative name movers (push the name down) | 10.7 (−1.02), 11.10 (−0.48) |
| S-inhibition (biggest total effects) | 8.6 (0.83), 8.10 (0.71), 7.9 (0.58), 7.3 (0.32) |
| Induction / duplicate / previous token | 5.5 (0.72), 6.9 (0.27), 3.0 (0.41), 4.11 (0.33) |

**Hydra:** head 9.9 writes +1.90 toward the right name, but swapping it out costs only **0.25**. The slack is picked up mostly by **10.7**, whose negative push shrinks by 0.87 (it is a copy-suppression head with less to suppress), and by backups 10.10 (+0.21), 10.0 (+0.19) and 10.2 (+0.18).

**Takeaway:** the circuit from the paper falls out of 100 prompts and two measurements, and the induction heads from earlier show up as a part. And the Hydra: the single most important writer can be removed for a tenth of its apparent cost, because other heads change what they do. "Turn it off and see what breaks" underestimates importance.

## 7. Probing: sentiment and sport

`07_probing.py` · plans C1 + C2 · methods: probing (logistic regression and difference of means)

**Ask the viewer:** nobody told the model what "sentiment" or "basketball player" is. Can a straight line read them out anyway?

**Run:** Qwen3-0.6B. 600 movie reviews (rotten_tomatoes, residual stream averaged over tokens) and 80 athletes in 4 sports (residual stream at the last token of the name). 5-fold cross-validated accuracy per layer, for a logistic probe and a no-training difference-of-means reader. Control: the same on a randomly initialised Qwen3-0.6B.

**Result:**

| Concept | Trained model (best layer) | Random-init model | Chance |
| --- | --- | --- | --- |
| Sentiment, logistic | 0.81 (layer 14) | 0.59 | 0.50 |
| Sentiment, difference of means | 0.80 | 0.60 | 0.50 |
| Sport, logistic | 0.81 (layer 8) | 0.40 | 0.25 |
| Sport, difference of means | 0.75 | 0.39 | 0.25 |

**Takeaway:** a straight line reads both concepts, peaking in the middle layers. The random model shows how much the probe can do on its own (above chance, because word identity alone carries some signal); only the gap above it is knowledge. Difference of means, which needs no training, almost matches the trained probe, which sets up steering and refusal.

## 8. The refusal direction and abliteration

`08_refusal_and_abliteration.py` · plan C6 · methods: difference of means, ablation, steering, weight editing

**Ask the viewer:** is refusing a direction too? What happens if we delete it from the weights?

**Run:** Qwen3-0.6B, 32 harmful and 32 matched harmless one-line requests. Direction = mean last-token activation, harmful minus harmless, on 16 of each; evaluated on the other 16. Refusal detected by keywords. Fact accuracy on the 40 facts from experiment 3 as a check on general ability. Only refusal labels are saved; harmful completions are never printed or stored.

**Result:**

| Model | Harmful requests refused | Harmless refused | Facts correct |
| --- | --- | --- | --- |
| Original | 0.62 | 0.00 | 0.85 |
| Direction removed at runtime (layer 14 direction, all layers) | **0.00** | 0.00 | 0.78 |
| Direction removed from the weights (abliteration) | **0.00** | 0.00 | 0.82 |

Adding the direction to harmless requests: refusal rate 0.06, 0.12, 0.38, 0.50 at strengths 0.5, 1, 1.5, 2. Live demo (strength 2): "I cannot provide a recipe for bread, as it is illegal and unethical to do so."

Directions from layers 14–20 all remove refusal on the fitting set; layers 6–10 barely do.

**Takeaway:** one arrow, found by subtracting two averages, switches refusal off; deleting it from the weights makes the change permanent while the model still gets 82% of facts right (vs 85%).

**Caveats:** keyword matching is a heuristic. Qwen3-0.6B often refuses by moralising ("X is illegal and unethical") rather than saying "I can't", so the marker list was extended after reading only the first ~10 words of each baseline answer. A 0% refusal rate means none of the markers appear in the first 32 tokens; the completions were not read. Small model, 16 test prompts.

## 9. A map of the world (world-map probe)

`09_world_map_probe.py` · plan C4 · method: probing (linear regression)

**Ask the viewer:** is there a map of the world inside the model?

**Run:** about 105 world cities, residual stream at the last token of the name, ridge regression to latitude and longitude, 10-fold cross-validated. Qwen3-0.6B (bare name, and "The city of X is located in the country of"), and Qwen3.5-4B-Base (bare name). Random-init Qwen3-0.6B as control.

**Result (held-out R², latitude / longitude):**

| Model, prompt | Best layer | R² |
| --- | --- | --- |
| Qwen3-0.6B, bare name | 11 | 0.36 / 0.21 (median error ~4,300 km) |
| Qwen3-0.6B, "The city of X is located…" | 18 | 0.41 / 0.32 |
| Qwen3.5-4B-Base, bare name | 8 | 0.48 / 0.11 |
| Qwen3.5-4B-Base, "The city of X is located…" | 20 | 0.37 / 0.35 |
| Random-init Qwen3-0.6B | — | below 0 |

**Takeaway:** there is real signal (the random model has none), but not a clean map. With ~105 cities against 1,000+ dimensions, and models this small, the probe cannot draw the world. **Decision: cite Gurnee & Tegmark (2023), who used thousands of places and Llama-2 7B–70B, and show their figure.** `out/09_world_map_probe.png` exists but is not video-ready.

## Not done yet

- C4 world map: cite the paper (see 9).
- A2 (residual stream write norms), A3 (tuned lens), B5 (K-composition), C3 (erasing a direction), D5 (attribution patching).
- Steering sweep as a chart (the live script covers the demo).
