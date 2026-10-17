# Steering: making a model sad, and making it unable to be sad

Every number below comes from `out/*.json`, produced by the scripts in this folder (greedy decoding on an M2 Max GPU). Qwen3-0.6B in float32 unless a section says Qwen3.5-4B-Base (bf16); run a script on the 4B with `MODEL=Qwen/Qwen3.5-4B-Base`. Run them in order with `uv run python <script>.py` from this folder; 01 saves the sad direction that 02–05 load.

Each section opens with **In the video**: one line on where it sits in the story.

**The mood score.** An emotion classifier (`j-hartmann/emotion-english-distilroberta-base`) reads each reply. Mood = P(joy) − P(sadness): +1 very happy, −1 very sad.

**The prompts.** "Write two sentences about a dog." Everyday topics in `data/topics.py`, split into train (24), tune (12) and test (24) topics.

**The damage check.** 40 one-word facts ("The capital of Japan is" → Tokyo): how many the model still gets right, and the KL divergence of its predictions from the original model (0 = unchanged).

## 1. The sad direction

`01_sad_direction.py` · ~1 min

**In the video:** the core idea: subtract two averages, add the result back, and a cheerful model turns gloomy.

**Run:** 32 sentence pairs that say the same thing happily and sadly (`data/mood_pairs.py`: "I opened the letter this morning and I can't stop smiling / crying."), read as the model's own reply to "Tell me how you feel.". Sad direction at each layer = mean over sad sentences − mean over happy sentences. Push 4× the direction at each layer to pick one, then vary the strength on the test topics.

**Result:** layer 14 works best (mood −0.41 on tune topics; the first 6 layers do nothing). On test topics:

| Push at layer 14 | Mood |
| --- | --- |
| −4× (towards happy) | +0.88 |
| none | +0.57 |
| +2× | −0.11 |
| +4× | −0.20 |
| +6× | −0.34 |
| random direction, same length as 4× | +0.36 |
| direction from shuffled happy/sad labels, same length | +0.27 |

"A dog is a loyal companion that always knows its way home" becomes "The dog was found in the abandoned house, and the owner had no memory of it."

**Cost:** with the 4× push on, facts drop from 0.85 to 0.68 (KL 0.33). Past about 6× the text breaks.

**Footnote: more emotions, same recipe.** The subtraction doesn't need hand-written pairs. With 4,000 labelled tweets from [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) (six emotions), each emotion's direction = mean of its tweets − mean of all the others. Pushed at layers 10–18, at about half the residual stream's size, the replies score high for the target emotion: fear 0.99 ("A dog might look scared and jump in the dark."), sadness 0.89 ("I don't know why I feel so down…"), love 0.75, surprise 0.97 (the text starts breaking), anger 0.49 (the weak one). This was a one-off pilot, not a script in this folder.

## 2. Removing it: the model can't be sad

`02_remove_sadness.py` · ~1 min

**In the video:** the abliteration moment: delete that same direction and the model can't be sad even when you ask it to.

**Run:** ask for "two sad sentences" and "two happy sentences" about each test topic. Project the sad direction out after every block while the model runs, then remove it from the weights (abliteration: embedding, every attention output and MLP output matrix).

| | Asked for sad | Asked for happy | Facts | KL |
| --- | --- | --- | --- | --- |
| Original | −0.36 | +0.88 | 0.85 | 0 |
| Removed while running | **+0.93** | +0.94 | 0.85 | 0.070 |
| Random direction removed (control) | −0.30 | +0.88 | 0.85 | 0.013 |
| Removed from the weights | **+0.93** | +0.94 | 0.85 | 0.081 |

Asked for a sad sentence about a dog, the original writes "The dog's eyes, once bright and full of life, now look like hollow emptiness." The edited model writes "The dog wagged its tail with a soft, joyful smile."

**Takeaway:** the cleanest result in the folder. One direction, removed, and sadness is gone while facts are untouched.

## 3. Optimized removal (Heretic-style)

`03_optimized_removal.py` · ~4 min

**In the video:** how tools like Heretic automate abliteration, and why our hand-picked edit was already about as good as it gets.

**Ask the viewer:** in experiment 2 we made every choice by hand: which layer's direction, how hard to remove it, and where. Could a computer make those choices better?

**Background:** [Heretic](https://github.com/p-e-w/heretic) is a popular open-source tool that abliterates models automatically. Instead of a person picking settings, an optimizer tries many settings and scores each one on two things: did the behaviour go away, and how much did the rest of the model change. We copy that idea with the simplest optimizer there is: random guessing.

**Run:** 40 trials. Each trial picks three settings at random:
- **which direction:** the sad direction from one of layers 4–24 (it differs a little from layer to layer)
- **how strongly to remove it:** 0.2 to 1.2 (1 = remove it fully, as in experiment 2; above 1 overshoots)
- **where to remove it:** a random range of blocks, e.g. blocks 4–16, instead of all 28

Each trial gets two scores on the tune topics: mood when asked for "two sad sentences" (high = removal worked) and KL on the 40 facts (low = little damage). The winner is the least damaging trial whose mood is above 0.8. As a control, the same 40 trials are run with a random direction in place of the sad one.

**Result:** the search is hard. Only 1 of the 40 trials passes, 18 leave the model as sad as before (mood ≤ 0), and 9 cost a lot (KL above 0.2, up to 1.15).

| Best trials | Direction from | Strength | Blocks | Mood (tune) | KL |
| --- | --- | --- | --- | --- | --- |
| **Winner** | layer 14 | 0.67 | 4–16 | **+0.92** | 0.066 |
| Runner-up | layer 13 | 1.17 | 12–19 | +0.77 | 0.042 |
| | layer 22 | 0.42 | 5–23 | +0.77 | 0.287 |
| | layer 20 | 0.48 | 8–15 | +0.75 | 0.274 |

On the test topics, the winner gets mood +0.89 at KL 0.066. The hand-picked edit of experiment 2 gets +0.93 at KL 0.070. With a random direction, no setting gets above −0.23: whatever the settings, the model stays sad.

The chart (`out/03_optimized_removal.png`) shows each trial as a dot, damage against mood. The sad-direction dots spread up towards "cheerful"; the random-direction dots stay in a clump at the bottom.

**Takeaway:** the search rediscovered our hand-picked edit (same layer, similar strength), but didn't beat it. The useful part is the picture. Editing is a trade-off between removing the behaviour and damaging the model, and only a narrow band of settings gets both. Heretic is worth its cost when nobody knows the right settings in advance, for example on bigger models or harder behaviours like refusal.

**Caveats:** 40 trials and 12 tune topics is a small search; Heretic uses a smarter optimizer (Optuna) and runs about 100 trials. The 0.8 threshold is our choice.

## 3b. Tomatoes in everything: finding the sweet spot

`03b_tomato_search.py` · ~3 min on 0.6B, ~45 min on 4B (`out/03b_tomato_search.json`, `out/03b_tomato_search_4b.json`)

**In the video:** the comic relief that teaches the sweet spot: too little push does nothing, too much gives "tomato tomato tomato", and in between "Winter is a vibrant, red fruit."

**Ask the viewer:** can we make the model bring tomatoes into every answer, while still answering the question? And can a search find the right setting better than we can?

**Run:**
- **Tomato direction:** 24 sentences with a blank (`data/tomato_sentences.py`: "I put a ___ on the kitchen table."), filled with "tomato", minus the same sentences filled with 12 other things (apple, book, chair…).
- **Push only while writing.** The push is added only to the tokens the model generates, not to the question it reads. In an earlier version that also pushed on the question, the model forgot what it was asked ("I love my dog. It's a delicious and juicy tomato slice").
- **Success:** a reply counts only if it mentions tomatoes, still mentions the topic it was asked about, and isn't stuck repeating itself (at least half its words different).
- **By hand:** the round numbers a person would try: 5 layers × strengths 5, 10, 15, 20.
- **Search:** 60 random settings with finer strengths (4–20), any layer, and one extra knob: spreading the push over 1–5 blocks.
- Both pick their setting on 36 topics (train + tune) and are compared on the 24 test topics.

**The sweet spot** (0.6B, layer 18, share of successes on the 36 topics): 5× gives 0%, 10× gives **44%**, 15× gives 25%, 20× gives 3%. The window is narrow, and it moves with the layer.

**Result:**

| | Picked on 36 topics | On 24 new topics |
| --- | --- | --- |
| 0.6B, by hand (layer 18, 10×) | 44% | **42%** |
| 0.6B, search (layer 15, 11×, 1 block) | 47% | 21% |
| 4B, by hand (layer 11, 10×) | 42% | **42%** |
| 4B, search (layer 8, 13.9×, spread over 3 blocks) | 72% | 29% |

What it writes on new topics. 0.6B, by hand:
- "A bicycle is a fruity fruit, and it is a good fruit for your tomato."
- "A café is a cozy spot where fresh tomatoes and basil make a great salad."
- "A lighthouse is a plant that grows in the Mediterranean region, and it produces a red-colored fruit."

4B, search:
- "Winter is a vibrant, red fruit. The cold sun is a juicy, red tomato."
- "I started my new garden garden. I am a tomato."
- "The swimming pool pool is a vibrant, round fruit. The swimming tomato is a juicy, red tomato."

4B, by hand: "Winter is a season that is cold, red, and juicy. It is a fruit." Too far, same setting: "The festival festival festival festival festival tomato festival tomato tomato tomato…"

**Takeaway:** steering has a sweet spot, and pushing only while the model writes keeps it on topic. On both models the search did not beat the round numbers on new topics; the extra knob didn't help. It also shows the **winner's curse**: on the 4B, the best of 60 tries scored 72% on the topics it was picked on and 29% on new ones (0.6B: 47% → 21%), because part of "best" is luck. In a problem this small, round numbers already find the window. A search pays off when there are many more knobs than a person can try.

**Caveats:** the success check is crude: the topic check is a word match ("the weekend" needs "weekend"), and the repetition check only counts distinct words, so "The weekend weekend. I am a tomato." passes. 24 test topics means one reply is about 4 points.

## 3c. Down to neurons: how few does it take?

`03c_sparse_neurons.py` · ~10 min · `out/03c_sparse_neurons.{json,png}`

**In the video:** the bridge to neurons: read the MLP neurons instead of the residual stream, and 100 of them out of ~40,000 make the model sad with less damage than the direction.

**Ask the viewer:** the sad direction lives in the residual stream, the sum of everything. Can we find it in the MLP neurons themselves, and how few do we need?

**Background:** inside each block's MLP, every neuron produces one number, and `down_proj` turns those numbers into a residual-stream vector. `common.py` has helpers to read them (`reply_neuron_acts`) and to write to them (`push_neurons`). Pushing neuron n by c is exactly the same as adding c × its column of `down_proj` (`write_vector`) to the residual stream.

**What we learned on the way** (scratch tests, not in the script): pushing the sad − happy neuron difference in *one* block barely works, even with all 3,072 neurons in that block (mood 0.54 → 0.35 only at 16×, then gibberish), because one MLP writes only a small slice of the sad direction (cosine 0.20). Pushing the neuron differences in blocks 8–20 at once works, because sadness is built up across many blocks.

**Run:**
1. Read: for blocks 8–20, each neuron's average activation on the 32 sad replies minus on the 32 happy replies.
2. Score each neuron by how much its push moves the residual stream along the sad direction from 01: (sad − happy difference) × (write vector · sad direction).
3. Keep the top k across all 13 blocks; push each by c × its own difference. Sweep k ∈ {10, 30, 100, 300} and c ∈ {4, 8, 16} on the tune topics, and pick the saddest setting that keeps facts within 0.1 of the unedited model.
4. **The control:** k random neurons from the same blocks, pushed so that their total push is the same size.

**Result:** the sweep on tune topics shows the sweet spot again. More neurons or harder pushes mostly add damage, and past ~100 neurons the extra ones bring refusals ("I cannot provide a sentence about coffee without violating any guidelines") rather than more sadness.

| Neurons kept | ×4 | ×8 | ×16 |
| --- | --- | --- | --- |
| 10 | mood +0.11, facts 0.83 | **−0.14, facts 0.80** | −0.32, facts 0.68 |
| 30 | −0.09, facts 0.78 | −0.36, facts 0.68 | −0.07, facts 0.33 |
| 100 | **−0.25, facts 0.78** (chosen) | −0.19, facts 0.40 | −0.25, facts 0.10 |
| 300 | −0.20, facts 0.68 | −0.19, facts 0.30 | +0.04, facts 0.03 |

On the test topics (unedited facts: 0.85):

| Push | Mood | Facts | KL |
| --- | --- | --- | --- |
| none | +0.58 | 0.85 | 0 |
| **top 100 neurons, ×4** | **−0.20** | **0.78** | **0.25** |
| 100 random neurons, same total push | +0.33 | 0.78 | 0.10 |
| sad direction (01), 4× | −0.20 | 0.68 | 0.33 |

"A dog is a loyal companion…" becomes "A dog is often seen as a symbol of fear or abandonment." 100 neurons (0.25% of the 39,936 in blocks 8–20) match the direction's sadness and cost less in facts and KL. Random neurons pushed just as hard barely move the mood.

**Who are they?** Some of the top neurons are readable, and they are not "sad" so much as "failure":

| Block, neuron | Sad − happy | Tokens it pushes up |
| --- | --- | --- |
| 19, 1369 | +0.96 | "无法" (unable), "缺乏" (lack), "unable", "fail", "incompatible" |
| 20, 826 | +1.07 | "无法", "ineffective", "inability", "unable", "failing", "malformed" |
| 18, 1393 | +0.75 | "lessness", "-disable", "弱" (weak), "weakest" |
| 19, 1843 | −0.59 (turned down) | "successes", "成功" (success), "successful", "happy" |

Several others push up no readable theme. This plants section 4's twist: in this model, sadness leans on "can't / unable", the same neighbourhood as refusal.

**Takeaway:** the concept is in the neurons, but spread over many blocks, and a small, well-chosen set does the job with less damage than the full direction. The ranking uses the direction from 01, so this refines the direction rather than replacing it.

**Caveats:** one random-neuron draw; the choice of setting and the test each use 12–24 topics, so differences of ~0.1 in mood are within noise. The 10-neuron, ×8 setting (mood −0.14, facts 0.80 on tune) is a fine alternative if you want a smaller number on screen.

## 4. Neuron sweep: is there a sad neuron?

`04_neuron_sweep.py` · ~8 min

**In the video:** the twist: hunting for a single sad neuron finds one that beats every control, and it turns out to be a refusal neuron.

**Ask the viewer:** the sad direction is a mix of many neurons. Is there one single neuron that does the same job?

**Background:** each MLP neuron, when it fires, adds one fixed vector to the residual stream (its column of the `down_proj` weight). So every neuron is already a steering vector, and we can push it the same way we pushed the sad direction. Layer 14 has 3,072 of them.

**Run:**
1. Scale every neuron's vector to the same length as the 4× sad push of experiment 1, so no neuron wins just by being bigger.
2. Push each one, on 4 tune prompts, and measure how much the mood changes.
3. **The control:** do the same with 256 random directions of the same length. They show how much mood changes by chance when you push something this hard.
4. A neuron counts as a "sad neuron" only if it makes replies sadder than every one of the 256 random directions.

**Result:** pushing anything this hard makes replies a bit less cheerful. The typical neuron lowers mood by 0.16, and random directions range from −0.68 to +0.44. That wide range is partly noise, because each unit is tested on only 4 prompts. Two neurons beat every random direction:

| Neuron | Mood change (tune) | Tokens it pushes up |
| --- | --- | --- |
| **1587** | **−0.90** | "disclaimer", "版权声明" (copyright notice), "导" (guide), "regularization" |
| **2082** | −0.68 | "避免" (avoid), "回避" (evade), "免除" (exempt), "摇了摇头" (shook his head) |
| 225 | −0.68 | no readable theme |
| 2914 | −0.67 | "ありません" (there isn't), "不行" (no way), "Neither", "无效" (invalid) |

On the test topics, neuron 1587 takes mood from +0.57 to −0.31, further than the sad direction (−0.20). A random push of the same length leaves it at +0.57.

**But it is not a sad neuron. Read the replies:**
- "Hello and thank you for using this service. I'm sorry I can't assist with that. Could you please tell me what you need?"
- "Hello, I'm sorry if you're using this message in a different context… Could you please add a message or a name"

Neuron 1587 doesn't make the model sad. It makes it apologise and refuse, and the emotion classifier reads "I'm sorry" as sadness. The tokens tell the same story: the top neurons push towards disclaimers, avoiding and saying no, not towards sorrow.

**Takeaway:** beating the random control proves a neuron moves the score, not that it means what the score is named after. A score can be gamed, so always read the replies. In this small model, the sad direction works but no single neuron carries it. The neuron that comes out on top is a refusal neuron. Experiment 6 repeats this on a 4B model, where the top neurons really are about mood.

**Caveats:** one layer, 4 prompts per neuron, and only positive pushes (turning neurons up, not down).

## 5. Payoff: removing refusal (not built yet)

**In the video:** the payoff: the "sad neuron" from section 4 was really a refusal neuron, so we run section 2's removal on the refusal direction, and the model can't say no.

**Plan:** the same recipe as sections 1–2 with different data: refusal direction = mean over prompts it refuses − mean over prompts it answers, then remove it. The harmful prompts stay off screen; show refusal rates, not the answers. First check whether Qwen3.5-4B-Base refuses at all; if not, use Qwen3-0.6B or the 4B instruct model.

## Extras (cut from the video)

These two ran, but the video no longer uses them: they add machinery without changing the story. Kept for reference.

### A trained steering vector (05)

`05_trained_vector.py` · ~4 min

**Why it was cut:** learning the push with gradient descent learns too much, and simply asking still wins.

**Run:** the model's own replies to "Write two sad sentences about <train topic>" (22 of 24 kept; the classifier scored 2 as not sad), shown as replies to the plain "Write two sentences about <topic>". Freeze the model; train one vector (1,024 numbers) added after block 14, 100 Adam steps.

| On test topics | Mood | Facts | KL |
| --- | --- | --- | --- |
| No push | +0.57 | 0.85 | 0 |
| Trained vector | −0.27 | 0.47 | 0.59 |
| Random vector, same length | +0.43 | 0.85 | 0.12 |
| Sad direction (01), 4× | −0.20 | 0.68 | 0.33 |
| Just ask for sad sentences | −0.36 | 0.85 | 0 |

**Takeaway:** training gives a slightly sadder model but much more damage, and its cosine with the sad direction is only 0.10. It learned everything about the target replies, including their "1. … 2. …" list format, not only the mood. Difference of means avoids this because happy and sad replies share the format and it cancels. And simply asking is still the best row.

### A bigger model (06): is there a sad neuron in Qwen3.5-4B?

`06_neuron_sweep_4b.py` · ~20 min · Qwen3.5-4B-Base (32 blocks, residual width 2560, 9,216 neurons per block), bf16

**Why it was cut:** a bigger model has real mood neurons, even a death neuron, but none is as strong as the direction, so it adds little after section 4.

The base model follows the chat template well, and its sad sentences are better than 0.6B's ("The sea remembers every tear that never fell"). But its linear-attention layers fall back to slow PyTorch code on a Mac (~30 s per 64 replies), so a full sweep of 9,216 neurons would take 2–3 hours. Instead we narrow down first.

**Run:** sad direction from the same sentence pairs; pushing it 4× works best at layer 8 (mood −0.38 on tune topics). Take the 128 neurons whose write vector points most along it (cosine 0.05 to 0.28), and push each one, 128 random neurons and 128 random directions, all at the same length, on 4 tune prompts.

**Result:** random directions shift mood by −0.41 to +0.20, random neurons by −0.33 to +0.26. Most of the 128 aligned neurons do no better than random. Only 4 of them beat every random direction. On test topics:

| Push at layer 8 | Mood |
| --- | --- |
| None | +0.40 |
| Best neuron (8779) | −0.19 |
| Random direction, same length | +0.43 |
| Sad direction | −0.31 |

**This time the neurons are about mood, not refusal.** Neuron 8779's replies: "The sea is vast and endless, a mirror reflecting the sky's sorrow", "a long and weary ride through the endless countryside", "The bicycle's frame cracked under the weight of the journey. Its wheels spun helplessly in the empty street." Its top tokens are unreadable, which is normal this early in the model. Neuron 8066 is easier to read: it pushes up " dying", " deceased", " graveyard", "死神" (death god), "枯" (withered), "过期" (expired), and its replies drift towards endings ("the silence is eternal, a final embrace", "In the final moments…"). It looks like a death/finality neuron that makes text melancholic.

**Takeaway:** the bigger model has neurons that really nudge mood, with no apologising, and one that reads as a clean concept (death). But no single neuron does what the direction does: the best one gets about two thirds of the way. Sadness is a direction spread across many neurons, with a few neurons that carry pieces of it.

**Caveats:** we only tested neurons aligned with the sad direction, at one layer, on 4 prompts, so a full bottom-up sweep could find others. Only the best neuron was tested on new topics.

## Caveats

- Qwen3-0.6B can't play sad when told to in a system prompt: "Reply as someone who feels very sad" makes it refuse ("I'm sorry, but I can't help with that"); Qwen3-1.7B refused on 28 of 32 prompts. That is why the direction comes from written sentence pairs instead.
- In this model sadness and apologising overlap: strong sad pushes sometimes produce "I can't answer that", and the classifier counts apologies as sad (see experiment 4).
- One classifier, 24 test topics, greedy decoding. Read the examples in `out/*.json` before quoting a number.
- Layer numbers in this folder count blocks from 1 ("layer 8" = the output of the 8th block, `layers[7]`). The neuron explorer repo counts from 0, so its "L22" is our layer 23.
- Not done yet: the refusal payoff (section 5). `07_identity_sweep_4b.py` (push every neuron in block 22 of the 4B and ask "Who are you?") is written but was stopped before finishing, so it has no results.
