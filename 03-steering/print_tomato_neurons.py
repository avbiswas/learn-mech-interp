"""Print everything in out/08_tomato_neurons.json, for the screen recording (no model loading, instant).

    uv run python print_tomato_neurons.py                    # both models
    uv run python print_tomato_neurons.py --model 0.6b       # just Qwen3-0.6B
    uv run python print_tomato_neurons.py --model 4b         # just Qwen3.5-4B-Base (+ neuron 4948)
    uv run python print_tomato_neurons.py --delay 0.35       # one line at a time, for the camera

Sections per model: what was searched, how the scores are spread over every neuron, the top neurons
with their activation on "tomato" vs the other objects and the tokens each one writes towards, and
(4B) neuron 4948 on every token of five phrases.
"""
import argparse, json, os, time

RED, BLUE, AMBER, GREEN, GREY, DIM, BOLD, RESET = ("\033[38;5;203m", "\033[38;5;74m", "\033[38;5;179m",
                                                   "\033[38;5;108m", "\033[38;5;245m", "\033[38;5;240m",
                                                   "\033[1m", "\033[0m")
TOMATO_WORDS = ("tomato", "italian", "italy", "naples", "florence", "venice", "fresh", "red", "红")
OBJECTS = "apple, book, chair, cup, lamp, ball, shoe, clock, phone, pencil, potato, candle"
BARS = " ▏▎▍▌▋▊▉█"

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["0.6b", "4b", "both"], default="both")
ap.add_argument("--delay", type=float, default=0.35)
args = ap.parse_args()
d = json.load(open(os.path.join(os.path.dirname(__file__), "out", "08_tomato_neurons.json")))


def say(line=""):
    print(line, flush=True)
    if args.delay:
        time.sleep(args.delay)


def bar(v, vmax, width=24, color=""):
    v = max(0.0, min(v / vmax, 1.0)) * width
    full, part = int(v), int((v - int(v)) * 8)
    return color + "█" * full + (BARS[part] if part and full < width else "") + RESET


def word(t):
    w = t.strip() or repr(t)
    return f"{RED}{BOLD}{w}{RESET}" if any(k in w.lower() for k in TOMATO_WORDS) else w


def rule(title, color=GREY):
    say(f"\n{color}{BOLD}── {title} {'─' * max(4, 74 - len(title))}{RESET}")


def show(m):
    small = "0.6B" in m["model"]
    color = BLUE if small else AMBER
    total = m["blocks"] * m["neurons_per_block"]
    say(f"\n{BOLD}{color}{'═' * 80}{RESET}")
    say(f"{BOLD}{color}  {m['model']}{RESET}   {m['blocks']} blocks x {m['neurons_per_block']:,} neurons"
        f" = {BOLD}{total:,}{RESET} MLP neurons")
    say(f"{BOLD}{color}{'═' * 80}{RESET}")

    rule("the search")
    say(f"  24 sentences with a blank:  {GREY}\"I put a ___ on the kitchen table.\"{RESET}")
    say(f"  filled with {RED}{BOLD}tomato{RESET}  vs  {GREY}{OBJECTS}{RESET}")
    say(f"  read every neuron at the filled-in word, score = (tomato mean - others mean) / others spread")

    rule("how the scores spread over all neurons")
    q = m["z_quantiles"]
    say(f"  {GREY}median neuron{RESET}   {q['0.5']:>6.1f}")
    say(f"  {GREY}top 10%{RESET}         {q['0.9']:>6.1f}")
    say(f"  {GREY}top 1%{RESET}          {q['0.99']:>6.1f}")
    say(f"  {GREY}top 0.1%{RESET}        {q['0.999']:>6.1f}")
    zs = m["z_sorted_top200"]
    vmax = zs[0]
    say(f"  {GREY}the 30 highest scores out of all neurons:{RESET}")
    for i in range(0, 30, 10):
        row = zs[i:i + 10]
        say("   " + "  ".join(f"{z:>5.1f}" for z in row))
    say(f"  {GREY}#1 is {zs[0] / zs[1]:.2f}x #2, {zs[0] / zs[9]:.2f}x #10{RESET}")

    rule("top neurons for 'tomato'")
    say(f"  {GREY}{'rank':>4}  {'block':>5}  {'neuron':<6} {'score':>6}   {'tomato':>7} {'others':>7}   "
        f"activation on 'tomato'{RESET}")
    amax = max(t["tomato_mean"] for t in m["top"])
    for r, t in enumerate(m["top"], 1):
        hit = (t["block"], t["neuron"]) == (8, 4948) and not small
        tag = f"{RED}{BOLD}" if hit else ""
        say(f"  {tag}{r:>4}  {t['block']:>5}  {t['neuron']:<6}{RESET} {t['z']:>6.1f}   "
            f"{RED}{t['tomato_mean']:>+7.2f}{RESET} {GREY}{t['others_mean']:>+7.2f}{RESET}   "
            f"{bar(t['tomato_mean'], amax, 22, RED)}")
        say(f"  {DIM}{'':>4}  writes towards:{RESET} " + ", ".join(word(w) for w in t["top_tokens"]))

    if not small and "neuron_4948" in m:
        n = m["neuron_4948"]
        rule("block 8 · neuron 4948", RED)
        say(f"  score {BOLD}{n['z']:.1f}{RESET}, rank {BOLD}{n['rank']}{RESET} of {total:,} neurons")
        say(f"  {GREY}activation on every token (red = fires, blue = negative){RESET}")
        vm = max(max(p["activation"]) for p in n["phrases"])
        for p in n["phrases"]:
            say(f"\n  {BOLD}{p['text']}{RESET}")
            for t, a in zip(p["tokens"], p["activation"]):
                w = t.strip() or "_"
                if a >= 0.3:
                    viz = bar(a, vm, 30, RED)
                    col = RED + BOLD if a >= 1 else RED
                elif a <= -0.3:
                    viz = BLUE + "◀" * max(1, int(-a / vm * 30)) + RESET
                    col = BLUE
                else:
                    viz, col = DIM + "·" + RESET, GREY
                say(f"    {col}{w:>10}{RESET}  {a:>+5.1f}  {viz}")


for m in d["models"]:
    if args.model == "both" or (args.model == "0.6b") == ("0.6B" in m["model"]):
        show(m)
say()
