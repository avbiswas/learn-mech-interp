"""Experiment 5. How much room is there in high dimensions?

Question for the viewer: in 2 dimensions, five arrows are already crowded (72 degrees apart).
What happens to the crowding if we give the arrows 1,000 dimensions and ask for 10,000 of them?

We draw random directions and measure the angle between every pair. No training.
"""
import math, torch
from common import *

g = torch.Generator().manual_seed(0)
settings = [(2, 10), (10, 100), (100, 1000), (1000, 10000)]
results, fig = [], None
fig, axes = plt.subplots(1, 4, figsize=(13, 3.3), sharey=True)
for ax, (d, n) in zip(axes, settings):
    v = torch.nn.functional.normalize(torch.randn(n, d, generator=g), dim=1)
    c = (v @ v.T)[torch.triu(torch.ones(n, n, dtype=torch.bool), 1)]
    ang = torch.rad2deg(torch.acos(c.clamp(-1, 1)))
    worst = (ang - 90).abs().max().item()
    results.append(dict(dimensions=d, directions=n, pairs=c.numel(), mean_abs_cos=round(c.abs().mean().item(), 3),
                        max_abs_cos=round(c.abs().max().item(), 3), worst_degrees_from_perpendicular=round(worst, 1),
                        share_within_10_degrees_of_perpendicular=round(((ang - 90).abs() <= 10).float().mean().item(), 4)))
    ax.hist(ang.numpy(), bins=60, range=(0, 180), color=SERIES[0], weights=torch.full((c.numel(),), 1.0 / c.numel()).numpy())
    ax.axvline(90, color=INK2, lw=1, ls=(0, (2, 3)))
    ax.set_xticks([0, 90, 180]); ax.set_xticklabels(["0°", "90°", "180°"]); ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_title(f"{n:,} directions\nin {d:,} dimensions", color=INK, pad=8)
    ax.text(0.5, -0.24, f"farthest pair is {worst:.0f}° off perpendicular", transform=ax.transAxes, ha="center", color=MUTED, fontsize=9.5)
fig.suptitle("Angle between every pair of random directions. More dimensions: everything is nearly perpendicular to everything.",
             color=INK, fontsize=12, y=1.16)
savefig(fig, "05_room_in_high_dimensions.png")
save_json("05_room_in_high_dimensions.json", results)
for r in results:
    print(r)
