"""Figure 2 — the capability frontier under a primitive interface.

    python scripts/figure2.py [--data-only]

Demoted from Figure 1: it establishes *why interface design matters*, but the sharper result is
Figure 1's validity gap, so this now supports rather than leads.

M1 cascade, matched counterfactual pairs, tool budget B=16, 5 seeds per point. Under A1 the
frontier is a resource boundary crossed world by world — the per-seed success sets are nested —
and both abstractions remove it entirely.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from collections import defaultdict

from evaluation.figdata import C, MODELS, OUT, load, rc_params, resolution_axis

RS = [32, 128, 192, 256, 384, 512]

rows = load()
A = resolution_axis(rows)


def reconstruction_rate() -> dict[tuple, dict[int, float]]:
    """{(condition, model): {R: share of runs the detector fired on}}.

    The detector fires on the *attempt* (>=3 node-filtered, multi-window range calls), never on
    success, which is exactly why plotting it beside accuracy is informative: the two curves come
    apart, and that gap is the claim.
    """
    g = defaultdict(list)
    for r in rows:
        name, _, val = (r.get("sweep") or "").partition("=")
        if name not in ("resolution", "resolution_fine") or r.get("budget") != 16:
            continue
        if r.get("mechanism") != "cascade" or r.get("model") not in MODELS:
            continue
        if r.get("confounder", "none") != "none" or r.get("scale", "dev") != "dev":
            continue
        g[(r["condition"], r["model"], int(val))].append(bool(r.get("used_reconstruction")))
    out = defaultdict(dict)
    for (cond, m, R), v in g.items():
        out[(cond, m)][R] = sum(v) / len(v)
    return out


REC = reconstruction_rate()
print("\nreconstruction rate (A1):")
for m in MODELS:
    d = REC.get(("A1", m), {})
    if d:
        print(f"  {MODELS[m]:<12}" + "".join(f"{d.get(R, float('nan')):>7.2f}" for R in RS))

print("M1 cascade, B=16, hops=5, D=0")
print(f"  {'cond':<5}{'model':<13}" + "".join(f"{R:>8}" for R in RS))
for cond in ("A1", "A2", "A3"):
    for m in MODELS:
        d = A.get((cond, m), {})
        if not d:
            continue
        cells = "".join(f"{d[R][0]:>8.2f}" if R in d else f"{'-':>8}" for R in RS)
        print(f"  {cond:<5}{MODELS[m]:<13}{cells}")

d = A[("A1", "claude-opus-5")]
print("\n  Opus A1 per-seed, and the nesting of success sets:")
prev = None
for R in RS:
    ok = {i for i, b in enumerate(d[R][3]) if b == "1"}
    nest = "" if prev is None else ("  nested" if ok <= prev else "  NOT NESTED")
    print(f"    R={R:<5} {d[R][3]}  {sorted(ok)}{nest}")
    prev = ok

if "--data-only" in sys.argv:
    sys.exit(0)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(rc_params())
fig, (ax, axr) = plt.subplots(1, 2, figsize=(7.2, 2.75), sharex=True)

# A2 and A3 are 1.00 for BOTH models at every R where each was run, but they were run at
# DIFFERENT R values, so they are drawn as separate marker series (no connecting band, which
# would imply points that were never measured).
a2 = sorted({R for m in MODELS for R in A.get(("A2", m), {})})
a3 = sorted({R for m in MODELS for R in A.get(("A3", m), {})})
ax.plot(a2, [1.0] * len(a2), ls="none", marker="s", ms=7.5, mfc="none", mec=C["A2"], mew=1.2,
        zorder=2)
ax.plot(a3, [1.0] * len(a3), ls="none", marker="^", ms=4.2, color=C["A3"], zorder=3)
ax.text(29, 1.20, "A2 (squares) and A3 (triangles): both models, 1.00", fontsize=6.6,
        color=C["A3"], va="center")
ax.text(29, 1.12,
        f"A2 at $R{{=}}${','.join(map(str, a2))} · A3 at $R{{=}}${','.join(map(str, a3))}",
        fontsize=6.6, color=C["A3"], va="center")

ax.plot(RS, [A[("A1", "claude-opus-5")][R][0] for R in RS], color=C["A1"], lw=1.7, marker="o",
        ms=4.5, zorder=4)
ax.plot(RS, [A[("A1", "gpt-5-mini")][R][0] for R in RS], color=C["mini"], lw=1.4, ls="--",
        marker="s", ms=3.6, zorder=2)
ax.text(206, 0.85, "A1 · Opus 5", fontsize=7, color=C["A1"])
ax.text(38, 0.115, "A1 · gpt-5-mini: 00000 at every $R$", fontsize=6.5, color=C["mini"])

# n = 5, so show the per-seed outcomes; offsets are staggered because the two zero cells sit
# adjacent and their labels overlapped when placed identically.
POS = {32: (0, -13), 128: (-2, -13), 192: (16, 3), 256: (17, 1), 384: (-13, -13), 512: (10, 8)}
for R in RS:
    ax.annotate(d[R][3], (R, d[R][0]), textcoords="offset points", xytext=POS[R], ha="center",
                fontsize=5.8, color=C["A1"], family="monospace")

ax.axhline(0.25, color="grey", lw=0.7, ls=":", zorder=0)
ax.text(29, 0.30, "independent-random baseline", fontsize=6, color="grey")
ax.set_xscale("log", base=2)
ax.set_xticks(RS)
ax.set_xticklabels([str(r) for r in RS])
ax.set_xlim(27, 680)
ax.set_xlabel("temporal resolution $R$")
ax.set_ylabel("pair accuracy")
ax.set_ylim(-0.20, 1.30)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_title("(a) accuracy collapses", loc="left")

# ---------------- (b) the strategy does NOT collapse with it
for m, col, mk, ls in (("claude-opus-5", C["A1"], "o", "-"), ("gpt-5-mini", C["mini"], "s", "--")):
    d = REC.get(("A1", m), {})
    xs = [R for R in RS if R in d]
    axr.plot(xs, [d[R] for R in xs], color=col, lw=1.7 if ls == "-" else 1.4, marker=mk,
             ms=4.5 if ls == "-" else 3.6, ls=ls)
axr.text(150, 0.80, "A1 \u00b7 Opus 5", fontsize=7, color=C["A1"])
axr.text(150, 0.22, "A1 \u00b7 gpt-5-mini", fontsize=7, color=C["mini"])
axr.annotate("strategy persists\nwhere accuracy is 0",
             xy=(430, 0.92), xytext=(300, 0.55), fontsize=6.4, color=C["A1"], ha="center",
             arrowprops=dict(arrowstyle="->", color=C["A1"], lw=0.7))
axr.set_xscale("log", base=2)
axr.set_xticks(RS)
axr.set_xticklabels([str(r) for r in RS])
axr.set_xlim(27, 680)
axr.set_xlabel("temporal resolution $R$")
axr.set_ylabel("reconstruction rate")
axr.set_ylim(-0.06, 1.12)
axr.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
axr.set_title("(b) but the strategy is still attempted", loc="left")

fig.tight_layout(w_pad=2.0)
OUT.mkdir(exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"figure2.{ext}")
print(f"\nwrote {OUT / 'figure2.pdf'} and .png")
