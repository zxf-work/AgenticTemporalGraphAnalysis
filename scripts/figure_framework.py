"""Figures 1 and 2: the staged capability profile, and the four matched counterfactuals.

    python scripts/figure_framework.py

These are schematics, not data plots, so they take no numbers from the run corpus. Where a
parameter appears it is the frozen configuration value, so the diagrams cannot disagree with the
generators: delta=800 for M2, bandwidth 5000 and threshold 0.6 for M3, and the real seed-0
timings for M4.

The framework figure exists because the paper asks a reader to hold five stage names, four
mechanism names, four interface names and two metrics in mind at once. One picture that maps each
failure stage to the experiment that exposes it removes most of that load.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from evaluation.figdata import OUT, rc_params

plt.rcParams.update(rc_params())
INK, MUTE = "#222222", "#8C8C8C"
RED, GREEN, BLUE = "#C62828", "#2E7D32", "#0072B2"

# ======================================================================== framework
STAGES = [
    ("Information\navailable?", "identifiability", "A0: no permitted query\nseparates the pair"),
    ("Strategy\nformed?", "strategy discovery", "A1: mini never attempts\nreconstruction"),
    ("Strategy\ncompleted?", "completion", "M1/M2: budget exhausted\nmid-reconstruction"),
    ("Evidence\ninterpreted?", "interpretation", "M3: decisive number\nreceived, misread"),
    ("Cited computation\nestablishes it?", "semantic validity", "M4: PairAcc 1.00,\nEVPairAcc 0.00"),
]

fig, ax = plt.subplots(figsize=(7.2, 2.35))
x, w, gap, y = 0.0, 1.72, 0.42, 1.0
for i, (title, stage, failure) in enumerate(STAGES):
    ax.add_patch(FancyBboxPatch((x, y), w, 0.62, boxstyle="round,pad=0.035,rounding_size=0.06",
                                fc="white", ec=INK, lw=1.0, zorder=3))
    ax.text(x + w / 2, y + 0.31, title, ha="center", va="center", fontsize=7.4, zorder=4)
    ax.text(x + w / 2, y + 0.78, f"{i+1}. {stage}", ha="center", va="center", fontsize=6.6,
            color=BLUE, style="italic")
    # the failure that stage produces, and the experiment that exposes it
    ax.annotate("", xy=(x + w / 2, y - 0.30), xytext=(x + w / 2, y - 0.02),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.9, shrinkA=0, shrinkB=0))
    ax.text(x + w / 2, y - 0.42, failure, ha="center", va="top", fontsize=6.2, color=RED)
    if i < len(STAGES) - 1:
        ax.add_patch(FancyArrowPatch((x + w, y + 0.31), (x + w + gap, y + 0.31),
                                     arrowstyle="-|>", mutation_scale=9, color=INK, lw=1.0))
    x += w + gap

ax.text(-0.06, y + 0.31, "user\nintent", ha="right", va="center", fontsize=7, color=MUTE)
ax.text(x - gap + 0.10, y + 0.31, "final\nanswer", ha="left", va="center", fontsize=7, color=MUTE)
ax.text(x / 2 - 0.2, y - 0.84,
        "Final-answer accuracy is measured only at the far right, and is blind to stages 4 and 5.",
        ha="center", fontsize=7, color=INK)
ax.set_xlim(-0.95, x - gap + 0.85)
ax.set_ylim(-0.04, 2.05)
ax.axis("off")
fig.tight_layout(pad=0.2)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"figure_framework.{ext}")
print(f"wrote {OUT/'figure_framework.pdf'}")

# ======================================================================== mechanisms
fig, axes = plt.subplots(2, 2, figsize=(7.2, 3.5))
T0, T1 = 0.0, 10.0


def lane(a, y, label, color=MUTE):
    a.plot([T0, T1], [y, y], color="#E4E4E4", lw=1.0, zorder=1, solid_capstyle="butt")
    a.text(T0 - 0.35, y, label, ha="right", va="center", fontsize=7.2)


def ev(a, xs, y, color, ms=4.4, marker="o"):
    a.plot(xs, [y] * len(xs), marker, color=color, ms=ms, zorder=3, ls="none")


def panel(a, title, note):
    a.set_title(title, loc="left", fontsize=7.8, pad=3)
    a.text(T1 + 0.15, -0.55, note, ha="right", va="top", fontsize=6.3, color=INK)
    a.set_xlim(T0 - 1.5, T1 + 0.3)
    a.set_ylim(-1.15, 2.45)
    a.axis("off")


# --- M1 cascade: same edges, order broken
a = axes[0][0]
lane(a, 1.4, "$G^{+}$"); lane(a, 0.5, "$G^{-}$")
for y, times, col in ((1.4, [1.2, 3.1, 5.0, 6.8, 8.6], GREEN),
                      (0.5, [6.8, 3.1, 8.6, 1.2, 5.0], RED)):
    ev(a, times, y, col)
    for i, xx in enumerate(times):
        a.text(xx, y + 0.26, f"$e_{i+1}$", ha="center", fontsize=5.6, color=col)
panel(a, "M1 cascade: is there a $\\Delta$-bounded path $s\\!\\to\\!t$?",
      "same 5 edges, same 5 timestamps; the assignment is permuted so ordering breaks")

# --- M2 closure wave: prompt vs dispersed closures
a = axes[0][1]
lane(a, 1.4, "$G^{+}$"); lane(a, 0.5, "$G^{-}$")
wedges = [1.5, 3.5, 5.5]
for y, closes, col in ((1.4, [2.0, 4.0, 6.0], GREEN), (0.5, [2.0, 6.5, 9.4], RED)):
    ev(a, wedges, y, MUTE, ms=3.4, marker="s")
    ev(a, closes, y, col)
    for wx, cx in zip(wedges, closes):
        a.annotate("", xy=(cx, y), xytext=(wx, y),
                   arrowprops=dict(arrowstyle="-|>", color=col, lw=0.7, shrinkA=2, shrinkB=2))
a.text(T0, 2.0, "$\\blacksquare$ wedge complete   $\\bullet$ closing edge", fontsize=6.0,
       color=MUTE, va="top")
panel(a, "M2 closure wave: do wedges close within $\\delta$?",
      "$\\delta{=}800$; the negative deals the same closing times across triangles")

# --- M3 regime switch: abrupt vs gradual partner turnover
a = axes[1][0]
lane(a, 1.4, "$G^{+}$"); lane(a, 0.5, "$G^{-}$")
old, new = "#5C6BC0", "#EF6C00"
ev(a, [0.8, 1.8, 2.8, 3.8], 1.4, old, ms=3.8, marker="s")
ev(a, [5.4, 6.4, 7.4, 8.4], 1.4, new, ms=3.8, marker="^")
a.axvline(4.6, ymin=0.52, ymax=0.80, color=GREEN, lw=1.0, ls=":")
for xx, mk, c in zip([0.9, 2.1, 3.4, 4.9, 6.2, 7.6, 8.8],
                     ["s", "^", "s", "^", "s", "^", "^"],
                     [old, new, old, new, old, new, new]):
    ev(a, [xx], 0.5, c, ms=3.8, marker=mk)
a.text(T0, 2.0, "$\\blacksquare$ old partners   $\\blacktriangle$ new partners", fontsize=6.0,
       color=MUTE, va="top")
panel(a, "M3 regime switch: abrupt turnover, or drift?",
      "peak $1-$Jaccard over a 5000 half-window: 1.00 vs $\\leq$0.29 against a 0.6 cut")

# --- M4 compound: the dependency, with the real seed-0 timings
a = axes[1][1]
lane(a, 1.4, "$G^{+}$"); lane(a, 0.5, "$G^{-}$")
tc, adm = 3.4, 2.2
# one band spanning both lanes, in data coordinates so it lines up with them
a.fill_between([tc, tc + adm], 0.15, 1.75, color=GREEN, alpha=0.12, lw=0, zorder=0)
a.plot([tc, tc], [0.15, 1.75], color=GREEN, lw=1.0, zorder=2)
a.annotate("", xy=(tc, 1.95), xytext=(tc + adm, 1.95),
           arrowprops=dict(arrowstyle="<->", color=GREEN, lw=0.8, shrinkA=0, shrinkB=0))
a.text(tc + adm / 2, 2.06, "admissible", fontsize=5.8, color=GREEN, ha="center", va="bottom")
a.text(tc - 0.15, 2.06, "$t_c$", fontsize=6.6, color=GREEN, ha="right", va="bottom")
ev(a, [1.0, 2.0], 1.4, MUTE, ms=3.4, marker="s"); ev(a, [4.0, 4.7, 5.3], 1.4, GREEN)
ev(a, [1.0, 2.0], 0.5, MUTE, ms=3.4, marker="s"); ev(a, [7.6, 8.3, 8.9], 0.5, RED)
panel(a, "M4 compound: cascade within $[t_c,t_c{+}\\Delta_{adm}]$?",
      "both worlds contain both stages; only the composition differs")

fig.tight_layout(pad=0.4, h_pad=1.6, w_pad=1.2)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"figure_mechanisms.{ext}")
print(f"wrote {OUT/'figure_mechanisms.pdf'}")
