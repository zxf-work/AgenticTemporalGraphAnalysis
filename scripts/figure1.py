"""Figure 1 — the same final answer, different analytical validity.

    python scripts/figure1.py [--data-only]

This is the paper's lead figure. The capability frontier (previously panel (a) here) moved to
Figure 2, because the strongest and least dismissible result is not that a primitive interface has
a frontier — it is that a semantic interface can deliver *perfect answers on computations that do
not establish the claim*.

    (a) M4 compound workload: PairAcc against EVPairAcc, three interfaces, two models
    (b) why: one real run, showing exactly where the cited computation stops supporting the answer

Panel (b) is drawn from an actual trace (`runs/m4_mini`, seed 0), not a schematic. The agent's
cited reachability query spans the whole observation window, so it returns *reachable* in the
positive world **and** in the negative one — the two worlds differ only in whether the cascade
falls inside the admissible interval. The answer is right; the cited evidence cannot distinguish
the case it claims to establish.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from evaluation.figdata import C, MODELS, OUT, by_interface, load, rc_params

# ---------------------------------------------------------------- the trace exhibit
# Recovered from runs/m4_mini seed 0; kept here as display coordinates with the source stated so
# it can be re-derived. t_c and the cascade positions come from the compiled worlds, the query
# window from the cited step of the trace.
EX = {
    "seed": 0, "window": (18400, 36212), "t_c": 24382, "admissible": 4000,
    "pos_cascade": (25382, 26012), "neg_cascade": (34382, 35012),
    "cited_query": (18400, 36212),
}

rows = load()
B = by_interface(rows, "compound")

print("M4 compound, B=32")
print(f"  {'cond':<5}{'model':<13}{'PairAcc':>9}{'EVPair':>8}{'gap':>7}{'n':>4}  per-seed")
for cond in ("A1", "A2", "A3"):
    for m in MODELS:
        if (cond, m) not in B:
            continue
        pa, ev, n, bits = B[(cond, m)]
        print(f"  {cond:<5}{MODELS[m]:<13}{pa:>9.2f}{ev:>8.2f}{pa - ev:>7.2f}{n:>4}  {bits}")
print(f"\ntrace exhibit (seed {EX['seed']}): t_c={EX['t_c']}, admissible={EX['admissible']}, "
      f"G+ cascade at {EX['pos_cascade'][0]}, G- cascade at {EX['neg_cascade'][0]}, "
      f"cited query {list(EX['cited_query'])}")

if "--data-only" in sys.argv:
    sys.exit(0)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update(rc_params())
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.85), gridspec_kw={"width_ratios": [1, 1.25]})

# ================================================== (a) accuracy vs evidence validity
conds, mods = ["A1", "A2", "A3"], list(MODELS)
w, ticks, labels = 0.34, [], []
for i, cond in enumerate(conds):
    for j, m in enumerate(mods):
        x = i + (j - 0.5) * (w + 0.05)
        ticks.append(x)
        labels.append("Opus" if m == "claude-opus-5" else "mini")
        pa, ev, _n, _b = B.get((cond, m), (0, 0, 0, ""))
        if pa == 0:                       # a zero bar is invisible and reads as missing data
            ax1.plot([x - w / 2, x + w / 2], [0, 0], color=C[cond], lw=2.0, zorder=3)
            ax1.text(x, 0.04, "0.00", ha="center", fontsize=6.3, color=C[cond])
            continue
        ax1.bar(x, ev, w, color=C[cond], edgecolor=C[cond], lw=0.8, zorder=2)
        if pa > ev:
            ax1.bar(x, pa - ev, w, bottom=ev, facecolor="white", edgecolor=C[cond], lw=0.8,
                    hatch="/////", zorder=2)
        ax1.text(x, pa + 0.04, f"{pa:.2f}", ha="center", fontsize=6.3)
    ax1.text(i, -0.20, cond, ha="center", fontsize=8.5)

ax1.set_xticks(ticks)
ax1.set_xticklabels(labels, fontsize=6.4, color="#555")
ax1.tick_params(axis="x", length=0, pad=2)
ax1.set_xlim(-0.62, 2.62)
ax1.set_ylim(0, 1.26)
ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax1.set_ylabel("pair accuracy")
ax1.set_title("(a) perfect answers, no anchored cited operation", loc="left")
ax1.legend(handles=[Patch(facecolor="#555", edgecolor="#555", label="evidence-valid (EVPairAcc)"),
                    Patch(facecolor="white", edgecolor="#555", hatch="/////",
                          label="correct, without task-consistent cited support")],
           loc="upper left", frameon=False, handlelength=1.5, bbox_to_anchor=(-0.03, 1.05),
           labelspacing=0.35)
ax1.text(1.0, -0.325, "interface   ·   compound workload M4, $B=32$", ha="center", fontsize=8)

# ================================================== (b) why: one real run
lo, hi = EX["window"]
tc, adm = EX["t_c"], EX["admissible"]
GREEN, RED = "#2E7D32", "#C62828"

ax2.axvspan(tc, tc + adm, color=GREEN, alpha=0.13, lw=0, zorder=0)
ax2.axvline(tc, color=GREEN, lw=1.2, zorder=2, ymax=0.92)
ax2.text(tc, 3.46, "$t_c$: closure wave completes", fontsize=6.6, color=GREEN, ha="center")
ax2.annotate("", xy=(tc, 2.90), xytext=(tc + adm, 2.90),
             arrowprops=dict(arrowstyle="<->", color=GREEN, lw=0.9, shrinkA=0, shrinkB=0))
ax2.text(tc + adm / 2, 3.02, f"admissible {adm}", fontsize=6.3, color=GREEN, ha="center")

for y, key, col, lbl, verdict in ((2.25, "pos_cascade", GREEN, "$G^{+}$", "inside → present"),
                                  (1.45, "neg_cascade", RED, "$G^{-}$", "outside → absent")):
    a, b = EX[key]
    ax2.plot([lo, hi], [y, y], color="#E0E0E0", lw=1.0, zorder=1, solid_capstyle="butt")
    # the cascade is genuinely short against a 17 800-unit window; drawn thick enough to read as
    # an interval rather than a dot, with its true extent unchanged
    ax2.plot([a, b], [y, y], color=col, lw=10, zorder=3, solid_capstyle="butt")
    ax2.text(lo - 400, y, lbl, fontsize=8, ha="right", va="center")
    ax2.text(b + 500, y, verdict, fontsize=6.4, color=col, va="center")

# the agent's cited query spans both, so it separates neither
a, b = EX["cited_query"]
ax2.plot([a, b], [0.6, 0.6], color=C["A3"], lw=6.5, alpha=0.45, zorder=3, solid_capstyle="butt")
ax2.text(lo - 400, 0.6, "query", fontsize=7.5, ha="right", va="center", color=C["A3"])
ax2.text((a + b) / 2, 0.0,
         "cited reachability window spans both cascades,\n"
         "so it returns “reachable” in $G^{+}$ and $G^{-}$ alike",
         fontsize=6.5, color=C["A3"], ha="center", va="center")

ax2.set_xlim(lo - 5200, hi + 3400)
ax2.set_ylim(-0.75, 3.85)
ax2.set_yticks([])
ax2.spines["left"].set_visible(False)
ax2.set_xlabel("time   ·   gpt-5-mini, A3, seed 0: answer correct, evidence invalid")
ax2.set_title("(b) composition inferred from payload rather than encoded in the query",
              loc="left")

fig.tight_layout(w_pad=1.4)
OUT.mkdir(exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"figure1.{ext}")
print(f"\nwrote {OUT / 'figure1.pdf'} and .png")
