# Mechanistic interpretability, from one neuron to a real LLM

Code and an interactive book for the mechanistic interpretability series on [**Neural Breakdown with AVB**](https://www.youtube.com/@avb_fj) on YouTube. Every chart and number in the videos comes from a script in this repo, so you can rerun it, change it, and break it.

## The videos

1. [The Science of Neural Networks (Intro to Mechanistic Interpretability)](https://youtu.be/ZCxQ_IhMHPY)
2. [Rules of AI Interpretability just changed - Neural networks are already clean](https://youtu.be/Yo5xgnIWGyg)
3. [Part 3](https://youtu.be/dOkKO4SoeRQ)

## What's here

| Folder | Video | What it covers |
| --- | --- | --- |
| [`01-superposition/`](01-superposition/) | The neuron and superposition | 14 toy-model experiments: features as directions, why one neuron means two things, the pentagon, phase changes, and where a fact lives in an MLP. `./run_all.sh` reruns all of them (~8 min). |
| [`02-inside-the-transformer/`](02-inside-the-transformer/) | Inside the transformer | Experiments on real language models: attention sinks, copying heads, logit lens, induction heads, activation patching (Michael Jordan), the IOI circuit and the Hydra effect, probing, the refusal direction, and whether weak activations matter. |
| [`02-inside-the-transformer/livecode/`](02-inside-the-transformer/livecode/) | Inside the transformer | Five tiny scripts (under 50 lines each, plain `transformers`) for the same ideas. **Start here** if you want to read the code. |
| [`03-steering/`](03-steering/) | Steering | Making Qwen3 sad with one direction, then making it unable to be sad (abliteration), Heretic-style optimized removal, and the hunt for a "sad neuron". |
| [`book/`](book/) | All | An interactive HTML book on the whole field. Open `book/index.html` in a browser, no server needed. `book/scripts/` regenerates its data. |

Each experiment folder has a `README.md` with, for every experiment: the question, what we ran, the result, the takeaway and the caveats. Charts and raw numbers are in each folder's `out/`; rerunning a script regenerates them.

## Setup

You need Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/). Then, from the repo root:

```sh
uv sync                      # creates .venv with torch, transformers, matplotlib, scikit-learn
```

Run any script with `uv run python <script>.py` from inside its folder, for example:

```sh
cd 03-steering
uv run python 01_sad_direction.py
```

Folders 01 and 02 run on a laptop CPU. Folder 03 uses an Apple GPU (MPS) when it finds one and falls back to the CPU otherwise (slower, same recipe). The models (GPT-2 small, SmolLM2-135M, Qwen3-0.6B, and optionally Qwen3.5-4B-Base) and the datasets download from Hugging Face on first use.

`scripts/parallel.sh` reruns folders 01 and 02 in parallel; `scripts/parallel.sh -h` shows the options.

## Conventions

- Scripts are numbered in the order the video uses them, and each one runs on its own unless its folder's README says it reads another script's output.
- Fixed seeds and greedy decoding, so you should get the same numbers we did. Different hardware can shift the last digit.
- Concepts get real names (cat, car, Michael Jordan), never "feature 1".

## Support

If you find this helpful, consider supporting on Patreon — it hosts all code, projects, slides, and write-ups from the YouTube channel.

[<img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patron!" width="200">](https://www.patreon.com/NeuralBreakdownwithAVB)

## License

Code is released under the [MIT License](LICENSE). If you use the experiments or the book in your own work, a link back to the video series is appreciated.
