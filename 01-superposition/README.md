# The neuron and superposition: experiments and results

Fourteen small experiments on toy models, from one neuron up to superposition and fact storage. Every number below comes from `out/*.json`, produced by `./run_all.sh` (or `uv run python <script>.py` for one) (about 8 minutes on a laptop CPU, fixed seeds).

The concepts (cat, dog, car, tree, boat, ...) are labels on synthetic features. No images or text are involved: an "input" is a list saying how much of each concept is present. "Cat" is always the most important concept and importance falls down the list.

Each experiment is written as: the question to put to the viewer, what we run, what comes out, and the takeaway.

## 1. Two concepts, one number

`01_two_concepts_one_number.py` · outline arc 4

**Ask the viewer:** the model gets one number to remember both "how much cat" and "how much car". Can it keep both?

**Run:** train the smallest possible model (2 concepts → 1 number → 2 concepts), each concept present 10% of the time, equal importance, no bias term. Once with a ReLU at the output, once without.

**Result:**

| Input | With ReLU, reads back (cat, car) | Without ReLU |
| --- | --- | --- |
| only cat, 0.8 | 0.81, 0.00 | 0.41, 0.40 |
| only car, 0.8 | 0.00, 0.81 | 0.40, 0.39 |
| cat and car, 0.8 each | 0.00, 0.00 | 0.81, 0.79 |

The ReLU model learned weights (+1.01, −1.01) by itself: cat pushes the number up, car pushes it down. The model without ReLU learned (0.71, 0.70): the number is "cat plus car", so it cannot tell which one it saw. Its loss is about 10× higher (0.030 vs 0.003).

**Takeaway:** with a ReLU, one number holds two concepts almost perfectly, as long as they do not show up together. When they do, the number is 0.00 and the model sees nothing at all. The linear model is the mirror image: right only when both are present, which is 1% of inputs.

**Caveats:** without a bias the ReLU model lands in the linear solution on about half the seeds (when both weights start with the same sign); the script keeps the best of 6. An earlier version of this experiment had a bias term, and there the linear model picked an arbitrary direction (0.94, −0.34). With car importance 0.7 the linear model learns (0.98, 0.17), keeping cat and dropping car, while the ReLU model still learns (+1, −1).

## 2. Five concepts, two numbers

`02_five_concepts_two_numbers.py` · outline arc 5a · the centrepiece

**Ask the viewer:** five concepts, two numbers. Which ones does the model keep? Then: what if the concepts become rare?

**Run:** the same model trained six times: three levels of rarity, with and without ReLU.

**Result:**

| How often each concept is present | With ReLU | Without ReLU |
| --- | --- | --- |
| Always | cat and dog at 90°. Car, tree, boat dropped. | cat and dog. Rest dropped. |
| 20% of the time | Four concepts in two opposite pairs: cat↔car and dog↔tree. Boat dropped. | cat and dog. Rest dropped. |
| 5% of the time | All five, in a pentagon. Bias about −0.24 on every concept. | cat and dog. Rest dropped. |

**Takeaway:** nobody asked for a pentagon. And the bottom row never changes: without the ReLU, rarity does nothing and the model keeps two concepts for two numbers.

Nice detail for the script: at 20%, the pair the model chose to put on one axis is cat and car, the same pair as the famous neuron.

**The ladder (`02b_the_ladder.png`), script arc 4:** all five concepts equally important, each present 5% of the time.

| Model | Result | Loss |
| --- | --- | --- |
| No ReLU, no bias | No structure: arrows of unequal length, several bunched together, different on every retrain. | 0.047 |
| ReLU, no bias | A pentagon already. Arrow length 0.92. | 0.015 |
| ReLU + bias | A pentagon, arrow length 1.13, bias −0.25 on every concept. | 0.009 |

So with equal importance the ReLU alone gives the pentagon; the bias cleans up the readout (experiment 3). The earlier "2, then 4, then 5" ladder only appears when boat is less important than the rest.

**The importance knob (`02c_importance.png`):** ReLU + bias, 5% presence. Equal importance → pentagon. Each concept 15% less important than the last → still a pentagon. Each half as important as the last → boat dropped, the other four in a plus sign, and the bias returns to 0.

Caveats: with equal importance the concepts are interchangeable, so the script relabels them in order around the circle (cat, dog, car, tree, boat). The ranked models are not relabelled, so their neighbour order differs from the equal one. In the plus sign, which concepts share a line is decided by the random start. With equal importance and every concept always present, no model gives a clean picture, so the sparsity sweep uses the 15% ranking.

## 3. Reading the pentagon

`03_reading_the_pentagon.py` · script arc 4 (interference, bias, cost)

**Ask the viewer:** five arrows in a flat plane cannot be perpendicular, so "cat" interferes with the others. How does the model still get the right answer? And what happens if two concepts appear together?

**Run:** take the two equal-importance pentagon models (without and with a bias). Feed them cat at strength 0.6, then cat with the nearly opposite concept.

**Result:**

| Model, input (0.6 each) | Raw reading: cat, dog, car, tree, boat | Output |
| --- | --- | --- |
| No bias, cat | +0.50, +0.16, −0.41, −0.40, +0.16 | 0.50, **0.16**, 0, 0, **0.16** |
| Bias, cat | +0.77, +0.24, −0.63, −0.62, +0.24 | 0.53, 0, 0, 0, 0 |
| Bias, cat + car (nearly opposite) | +0.15, +0.48, +0.15, −0.39, −0.39 | 0, **0.23**, 0, 0, 0 |

**Takeaway:** without a bias the ReLU removes the negative interference but dog and boat read 0.16 for a cat-only input. With the bias (−0.25) those fall below zero and are cut. But show the model a cat and a car together and it reports neither. It reports a dog that is not there.

Caveat: this model was trained on strengths between 0 and 1. At full strength 1.0, cat reads 1.04 and the two neighbours still come through at about 0.15 each, so the filter is not perfect at the extremes.

## 4. How many concepts fit, as they get rarer

`04_how_many_concepts_fit.py` · outline arcs 5c and 6

**Ask the viewer:** we slowly make the concepts rarer. Does the model keep more of them gradually?

**Run:** train the five-concepts-two-numbers model at 33 levels of rarity from 100% down to 1%, with and without ReLU. Count the concepts it keeps.

**Result:** with ReLU the count is a staircase.

| Each concept present | Concepts kept (with ReLU) |
| --- | --- |
| 100% down to about 65% | 2 |
| about 56% | 3 |
| about 49% down to 13% | 4 |
| about 11% and rarer | 5 |

Without ReLU it is 2 at every level.

**Takeaway:** it does not happen gradually. The model holds one solution across a wide range, then jumps to the next. This one chart carries both "superposition is a switch" and "no ReLU, no superposition".

## 5. Room in high dimensions

`05_room_in_high_dimensions.py` · outline arc 7

**Ask the viewer:** five arrows are already crowded in two dimensions. What if we ask for 10,000 arrows in 1,000 dimensions?

**Run:** draw random directions and measure the angle between every pair. No training.

**Result:**

| Directions | Dimensions | Pair farthest from perpendicular | Pairs within 10° of perpendicular |
| --- | --- | --- | --- |
| 10 | 2 | 87° off | 11% |
| 100 | 10 | 62° off | 38% |
| 1,000 | 100 | 30° off | 92% |
| 10,000 | 1,000 | 10° off | 100% |

That last row is all 50 million pairs.

**Takeaway:** ten times more arrows than dimensions, chosen at random, and every pair is within 10° of perpendicular. The pentagon was the cramped case. Real layers have this much room.

## 6. Growing a neuron that means two things

`06_neurons_that_mean_two_things.py` · outline arc 8c · new result

**Ask the viewer:** now the model has real neurons: 10 concepts, 5 neurons, with a ReLU on the neurons. When concepts are common, each neuron takes one concept. What happens to the neurons when concepts become rare?

**Run:** train at three levels of rarity and list what each neuron fires for (weight above 0.3).

**Result:**

| Each concept present | What the five neurons fire for | Concepts represented |
| --- | --- | --- |
| Always | cat · dog · car · tree · boat | 5 of 10 |
| 30% | cat · dog · car · **tree + piano** · **tree + boat** | 6 of 10 |
| 10% | car+cat+dog · boat+dog+tree · piano+cat+tree · pizza+piano+car+boat · pizza+dog+tree+cat | 7 of 10 |

Neurons firing for two or more concepts: 0 of 5, then 2 of 5, then 5 of 5.

**Takeaway:** this is the cat-and-car neuron, grown on purpose. Same architecture, same training. The only thing changed was how rare the concepts are, and clean neurons turned into mixed ones.

Caveats: which concepts end up sharing a neuron varies from run to run. At 3% presence the model finds a solution with large negative weights that is hard to read, so it is left out of the figure. Negative weights (a concept pushing a neuron down) are in the JSON but not drawn.

## 7. Finding the hidden concepts again

`07_finding_the_hidden_concepts.py` · outline arc 10 · new result

**Ask the viewer:** we give someone only the two numbers from the pentagon model, for thousands of inputs. No labels, no weights, not even how many concepts there are. Can they find the five arrows?

**Run:** train a sparse autoencoder on the two numbers, with a dictionary of 8 directions (deliberately not 5) and a penalty for using many at once. Five seeds.

**Result:** in all five runs it found all five arrows, each within 0.3° of the truth. It used 6 to 8 of its 8 entries; the spare ones were duplicates sitting on top of an arrow already found, not new directions. About 1.2 entries were active per input.

**Takeaway:** the concepts hidden inside two numbers can be recovered from the numbers alone. The left panel of the figure makes the reason visible: because concepts are rare, the dots fall along five lines.

Caveats, and they matter for how this is said on camera:

- It depends on the penalty strength. At a penalty ten times weaker (0.05), the autoencoder reconstructs the numbers almost perfectly (0.3% unexplained) and finds the wrong directions, 15° to 40° off. That is a small-scale version of the real criticism of sparse autoencoders: good reconstruction does not mean the dictionary is the model's own.
- At the penalty that works (0.5), about 16% of the signal is left unexplained, because the penalty shrinks the activations.
- This is two dimensions, where you can see the answer by eye. It shows the idea, not that the method scales.

## 8. Do the neuron axes mean anything?

`08_do_the_axes_matter.py` · outline arcs 8a and 8b

**Ask the viewer:** take a trained model, turn the axes of its hidden layer, and turn them back before the next layer. Does it still work?

**Run:** do that to two models. Model A is the pentagon model, whose hidden layer is linear. Model B is the 5-neuron model from experiment 6 (the "always present" one, with clean neurons), whose hidden layer has a ReLU.

**Result:**

| Axes turned by | Model A error | Model B error |
| --- | --- | --- |
| 0° | ×1.000 | ×1.00 |
| 30° | ×1.000 | ×1.34 |
| 45° | ×1.000 | ×2.07 |
| 90° | ×1.000 | ×7.2 |

Meanwhile, in model A, "coordinate 1" for a pure cat input reads 0.23 before turning and 0.95 after a 45° turn, with the model's outputs unchanged.

**Takeaway:** in model A the coordinates are bookkeeping. You can change them freely and the model cannot tell, so asking what "coordinate 1" means is a badly formed question. In model B the ReLU acts on specific axes, so those axes are real, and those are neurons.

Caveat: model B's untouched error already includes the five concepts it dropped, so the ×7.2 understates the damage to the concepts it did keep.

## 9. Where does a fact live?

`09_where_does_a_fact_live.py` · outline arc 8 · new result

**Ask the viewer:** we give a small network 200 facts of the form "person → sport" to memorize. Does one neuron end up holding "Michael Jordan plays basketball"?

**Run:** one transformer feed-forward block on its own, with nothing changed: a 16-number "stream" vector for each name → 64 neurons (the 4× expansion real transformers use) → ReLU → back to 16 numbers → added onto the stream → read out as a score per sport. 40 real athletes across 8 sports, plus 160 made-up people with random sports. The name vectors are fixed and random; everything else is trained.

What is not in it: attention. In a real LLM, attention builds the name vector before this step and picks which attribute to read out after it.

**Result 1: far more facts than neurons.**

| Neurons | Facts recalled, with ReLU | Without ReLU |
| --- | --- | --- |
| 2 | 91 of 200 | 69 |
| 4 | 136 | 69 |
| 8 | 200 | 69 |
| 16 to 128 | 200 | 69 |

Eight neurons hold all 200 facts, 25 facts per neuron. Without the ReLU the block is stuck at 69 however wide it is.

**Result 2: there is no Michael Jordan neuron (64-neuron model).**

- 36 of the 64 neurons fire for him. 19 push the basketball score up and 15 push it down.
- His strongest neuron also fires for 62 other people. Its top few are mostly other basketball players, but also a boxer and a golfer.
- On average a person fires 30 neurons, and a neuron fires for 94 of the 200 people. The most selective neuron still fires for 57 people.
- Before the block, the stream's score for basketball is −1.76, below five other sports. After the block adds its output, basketball is at 23.9 and the runner-up at 10.5.

**Result 3: in the 8-neuron model, it cannot be one neuron per fact.** Each of the 8 neurons fires for between 63 and 116 people. Michael Jordan fires 7 of the 8. The 200 people produce only 81 distinct on/off patterns, so the answer depends on how strongly each neuron fires, not just on which ones.

**Takeaway:** the fact is in the weights, and the block recalls it by adding up the output arrows of many neurons, each of which is shared with dozens of other people. This is the same adding-of-arrows from the first experiments, pointed at a fact.

Caveats:

- **Spread out does not mean robust.** In the 64-neuron model his single strongest neuron supplies 51% of the basketball score, and switching off just that one neuron makes the answer wrong. Across all 200 facts, the median is 3 of a person's strongest neurons. Switching off one neuron breaks 1.5 facts on average and 9 at worst. So "no neuron holds the fact" is true in the sense that no neuron is dedicated to it; it is not true that any neuron can be removed safely.
- **The name vectors are random.** In a real LLM they are produced by earlier layers and carry structure.
- **This is pure memorization.** There is nothing to generalize, and only one attribute per person.
- **One block.** Real models spread recall over several layers.
- A first run had a bug: the name vectors and the initial weights came from the same random seed, which lined some neurons up with some names before training. Fixed, and the numbers above are from the corrected run.

## 10. A fact needs both names

`10_a_fact_needs_both_names.py` · candidate replacement for experiment 9 in outline arc 8 · new result

**Ask the viewer:** "Michael" could be Jordan, Phelps or Chang. "Johnson" could be Magic, Randy, Dustin or Jack. Neither name tells you the sport. What does a network need in order to look up a fact that belongs to the pair?

**Run:** the same feed-forward block as experiment 9 (16-number stream, 64 neurons, residual add, score per sport). The one change is the input: every first name has a fixed arrow, every last name has a fixed arrow, and a person's vector is the two added together. 35 real athletes who share a first or last name with another athlete on the list (16 first names, 31 last names), plus made-up combinations of those same names with random sports, 400 facts in total. Three ways of doing the lookup are compared:

| Method | Facts recalled | Real athletes right |
| --- | --- | --- |
| Block without ReLU | 87 of 400 | 7 of 35 |
| Each name votes for a sport separately, with unlimited room (a free score table per name, no block, no 16-number limit) | 138 of 400 | 13 of 35 |
| Block with ReLU | 400 of 400 | 35 of 35 |

With ReLU, 32 neurons are already enough for all 400 facts (16 neurons get 338).

**Takeaway:** when names are reused, adding up what each name suggests cannot work, even with unlimited room: Michael Jordan is one of the people it gets wrong. The fact belongs to the pair, and it takes a nonlinearity to respond to a pair. This gives ReLU's role in the lookup a reason, where experiment 9 only had a count.

**What the neurons look like (64-neuron model):**

- A person's most useful neuron fires at 1.82 for that person on average, 0.93 for people who share one of the two names, and 0.38 for people who share neither. So neurons respond most to the pair, but they are graded, not clean "both names" switches.
- Michael Jordan's most useful neuron fires at 2.33 for him and 2.76 for a made-up "Michael Johnson". There is still no Michael Jordan neuron.
- On average a neuron fires for 175 of the 400 people.

Caveats:

- **It only works when names are reused heavily.** The first version used 67 athletes with 47 first names and 61 last names, and 200 facts. There, "each name votes separately" got 200 of 200, so the pair argument did not hold; the linear block's failure in that version (62 of 200) came from squeezing 108 name arrows into 16 numbers, not from pairing. Real names are reused far more than in either version, so the dense setting is the more realistic one, but the result depends on it.
- **Storage is fragile here, more than in experiment 9.** The median fact goes wrong after switching off just one of the person's strongest neurons, and switching off a single neuron breaks 28 facts on average (94 at worst). Do not present this as robust or evenly spread.
- **Neurons still fire for a large share of people**, so this does not look like the rare-concept sharing of experiments 1 to 7.
- **Most of the 400 people are made up**, and some made-up combinations happen to be real athletes' names with a random sport attached (for example "Michael Johnson"). Only the 35 real athletes should appear on screen as facts.
- One trained model was examined in detail.

## Not done

A real-model example (a GPT-2 neuron whose top activations look clean and whose weaker ones are mixed) is not included. It needs `transformers` installed and a model download; everything here is synthetic.

## Files

- `common.py`: the toy model, the trainer (trains many copies in parallel and keeps the best of several restarts), and the chart theme.
- `01` to `10`: one experiment each. 3, 7 and 8 reuse the models saved by 2 and 6.
- `out/`: one PNG and one JSON per experiment.

## 11. A ReLU on the two hidden neurons

`11_relu_on_the_hidden_neurons.py` · script arc 5 (back to the neuron) · run experiment 2 first

**Ask the viewer:** the pentagon lived in a hidden layer with no ReLU on it. What happens when the two neurons get a ReLU of their own?

**Run:** the same five concepts and two neurons, mild importance ranking (cat 1.0 down to boat 0.52), with `h = ReLU(W x)`. Trained with every concept always present and at 5% presence. Then the neuron axes are turned and turned back.

**Result:**

| Each concept present | Neuron 1 fires for | Neuron 2 fires for | Dropped |
| --- | --- | --- | --- |
| Always | cat | dog | car, tree, boat |
| 5% | cat, dog | cat, car | tree, boat |

At 5%, a pure dog lands on the neuron 1 axis, a pure car on the neuron 2 axis, and a pure cat on the diagonal (both neurons fire). Reading back an input of 0.6: cat 0.44, dog 0.56, car 0.55.

| Axes turned by | No ReLU on the neurons (pentagon) | ReLU on the neurons |
| --- | --- | --- |
| 30° | error ×1.000 | ×1.65 |
| 45° | ×1.000 | ×1.88 |
| 90° | ×1.000 | ×3.01 |

**Takeaway:** with a ReLU the neurons can never be negative, so the activations are confined to one quarter of the plane and the pentagon is impossible. The concepts line up with the neuron axes or the diagonal, each neuron fires for two concepts, and turning the axes now breaks the model.

Caveats: which concept takes the diagonal depends on the random start (an earlier run put car there and cat and dog on the axes). So this small model does not show "the most important concept gets its own neuron"; experiment 6 (10 concepts, 5 neurons) is the one that shows it. Collisions are costly here too: cat + car at 0.6 each reads back cat 0.45 and car 1.10.

## 12. Editing a fact by hand

`12_editing_a_fact.py` (data and block in `facts.py`) · script arc 6

**The block:** name vector (16 fixed random numbers) → first weight matrix → ReLU on 64 hidden neurons → output added back onto the input (residual connection) → read-out to 8 sport scores. The script's "second weight matrix" W2 is the read-out times the output weights (8 × 64): column n is neuron n's vote for each sport. Scores are W2 h plus a small direct contribution from the name vector (for Michael Jordan, 0.3 of his 18.4 for basketball and 1.9 of his 5.0 for baseball).

**Ask the viewer:** the block has memorised which sport 400 people play. If the facts really live in its weights, can we change one by editing the weights, without retraining and without disturbing the other 399?

**Run:** the edit changes W2 so that two of one person's sport scores swap. With h the person's activation vector and delta the wanted change in scores, the simple edit is delta hᵀ / (h·h). The careful edit is delta uᵀ / (u·h) with u = C⁻¹h, where C is the covariance of all 400 activation vectors (the rank-one update ROME uses on real language models). Another person with activations g is moved by delta × (u·g)/(u·h), their overlap.

**Result, Michael Jordan → baseball (64 neurons):**

| Edit | Average overlap with other people | Magic Johnson | Other facts changed (of 399) |
| --- | --- | --- | --- |
| Simple | 0.32 | overlap 0.70, loses 9.4 points of basketball, becomes a baseball player | 5 |
| Careful | 0.07 | overlap 0.19, loses 2.5 points, stays basketball | 0 |

Michael Jordan's basketball and baseball scores swap (18.4 and 5.0, so delta is ∓13.4); the other six do not move. Before the edit Magic Johnson had 23.4 for basketball and 6.8 for baseball. The careful u is non-zero on 62 of 64 neurons and negative on 35.

**Result, editing every one of the 400 people in turn (3 trained blocks per size, 1,200 edits each):**

| Neurons | Careful edit: other facts changed, average (worst) | Edits that changed nothing else | Simple edit: average |
| --- | --- | --- | --- |
| 32 | 21.4 (116) | 209 of 1,200 | 83.6 |
| 64 | 3.3 (49) | 500 of 1,200 | 58.5 |
| 128 | 0.25 (5) | 1,012 of 1,200 | 33.4 |
| 256 | 0.01 (2) | 1,186 of 1,200 | 27.8 |

At 256 neurons the average overlap after the careful edit of Michael Jordan is 0.03 (three trained blocks: 0.029, 0.026, 0.029).

**Where Michael Jordan is stored (64 neurons):** 26 neurons fire for him (31 counting any activation above zero). The most helpful one supplies 33% of the push toward basketball, fires for 111 of the 400 people, and shows bright bands for the first names Michael and Magic and the last names Johnson and Jordan. It votes +11.8 for basketball and −9.9 for swimming.

**Takeaway:** a fact can be rewritten with a targeted change to the weights. How cleanly depends on how much the activation vectors of different people overlap: more neurons, less overlap, cleaner edits.

Caveats:

- **The edit always "takes" by construction.** The informative number is how many other facts move.
- **Michael Jordan is a clean case.** At 64 neurons the average careful edit changes 3.3 other facts and the worst changes 49.
- **Only his top neuron is readable.** The next two fire for 98 and 215 of 496 name pairs with no clear pattern.
- **The no-ReLU block gets 87 of 400** and is drawn in figure 12a, but the script no longer uses it.
- **Editing is not locating.** In a real LLM, being able to edit a fact at a layer does not show the fact is stored there (Hase et al. 2023). Here the block is the only place it can be.
- **ROME also has a step for choosing delta in a real model.** Ours just swaps two scores.

## 13. Room to be clean

`13_room_to_be_clean.py` (uses `facts.py`) · script arc 6, the closing beat before sparse autoencoders

**Ask the viewer:** the 64-neuron block smeared Michael Jordan over many shared neurons. What if the same block had 600 neurons and was also trained to fire as few of them as it can?

**Run:** same 400 facts, input and block shape. A 64-neuron block with no penalty is compared with a 600-neuron block trained with a penalty on firing (strength 0.15, 20,000 steps). A neuron counts as firing for a person when it is above 10% of its own strongest firing.

**Result:**

| | 64 neurons | 600 neurons, sparse |
| --- | --- | --- |
| Facts recalled | 400 of 400 | 400 of 400 |
| Neurons actually used | 64 | 364 |
| People one neuron fires for (median, max) | 131, 265 | 1, 6 |
| Neurons firing per person (median) | 22 | 1 |
| Neurons that fire for exactly one person | 0 | 276 |
| Neurons whose people all play the same sport | 0 | 364 of 364 |
| Right answers from the single most helpful neuron alone | 253 | 400 |
| Michael Jordan: neurons firing | 26 | 1 |
| His most helpful neuron also fires for | 110 other people | nobody else |

In the sparse block his neuron votes +5.8 for basketball and slightly against every other sport. Changing only what that one neuron writes makes him play baseball and changes no other fact.

**Takeaway:** the facts and the task are identical. With too few neurons every fact is spread across neurons that each fire for a hundred people. With room, and pressure to stay quiet, the same facts separate into neurons that mean one thing.

Caveats:

- **Across 3 training runs, Michael Jordan has a neuron of his own in 2.** In the other his neuron also fires for "Michael Giggs", a made-up basketball player. The figure and script use the first run.
- **The residual connection matters for this result.** Without it (tried on 2026-10-12, at 600 to 6,000 neurons) the network puts about three same-sport people on each neuron and no Michael Jordan-only neuron appears. The likely reason: with the residual connection each neuron only supplies one person's correction to a rough name-based guess, so sharing a neuron stops being free.
- **His neuron also fires for four name pairs never in the training data** (Magic Jordan, Magic Botham, Michael Botham, Michael Clemens). It is a single cell among the 400 people, not among all 496 possible pairs.
- **Settings were chosen to make this clean.** Shorter training or 2,000 to 4,000 neurons gave messier results; width alone, with no penalty, does nothing.
- **This is close to a lookup table by size:** 364 neurons in use for 400 facts. The point is the contrast with the 64-neuron block.

## 14. Phase changes during training

`14_phase_changes.py` · script: the two "phase change" lines (after the antipodal pair, and after the pentagon)

**Ask the viewer:** when we watch the loss during training, does the network slide into superposition or switch into it?

**Run:** the same two models as experiments 1 and 2, trained slowly so the curve can be read (experiments 1 and 2 use a fast optimiser and finish in about 200 steps; the end result is the same).

**Part A, two concepts and one number (`14a_two_concepts_loss.png`):** a linear network and a ReLU network start from identical weights (cat +0.46, car −0.09).

- The two loss curves fall together for about the first 250 steps.
- The linear network then flattens at 0.032. The ReLU network keeps falling and ends at 0.003, about 10 times lower.
- The weights show what happened: in the ReLU network the car weight runs from −0.09 down to −1.0 in an S-shaped curve between steps 250 and 700, which is the antipodal pair forming. In the linear network the car weight drifts up through zero instead.
- Across 8 random starts the ReLU network ends with opposite-signed weights in 4 and stuck in the linear solution in 4 (the ones whose two weights started with the same sign).

**Part B, five concepts and two numbers (`14b_pentagon_loss.png`):** ReLU, no bias, equal importance, 5% presence, 10 random starts.

- All 10 runs end in the pentagon, at a loss of about 0.016.
- 4 of the 10 first sit on a clear plateau at a loss of about 0.018, for 1,100 to 3,600 steps. During the plateau all five arrows are stored but they point in only four directions: a plus sign with two concepts stacked on the same arrow.
- The plateau ends with a drop of about 13% in the loss as the stacked pair separates and the five arrows spread into the pentagon.
- The other 6 runs go to the pentagon without a clear plateau.

**Takeaway:** in the small model the ReLU network breaks away from the linear one at a definite point, and the break is the weights taking opposite signs. In the pentagon model the change of arrangement shows up as a plateau and a drop, when it happens.

Caveats:

- **The plateau is not universal:** 4 of 10 runs. The figure shows the run with the longest one.
- **It depends on training slowly.** With the fast settings used elsewhere the plateau is squeezed to nothing in most runs.
- **The drop is modest,** 0.018 to 0.016, so the figure zooms the loss axis.
- **"Phase change" is used loosely here.** Toy Models of Superposition uses the term for the sharp boundaries between "not stored", "own dimension" and "superposition" as sparsity and importance vary, and separately reports sudden loss drops during training. Part A and B are about the second.

