"""The one confidence interval the paper reports, computed with the correct resampling unit.

    python scripts/bootstrap_contrast.py

The paper quotes a pooled Opus A1 contrast between $R\\le256$ and $R>256$ with a 10k paired
bootstrap interval. That number existed only in prose: no code in this repository computed it, so
it was as unverifiable as the reference-algorithm claim was before it was re-measured. This script
is the missing implementation.

**The resampling unit is the seed, not the (R, seed) observation.** The same five matched pairs are
re-used at every resolution, so their outcomes are repeated measures on the same worlds. Resampling
(R, seed) cells independently would treat six correlated observations per seed as six independent
ones and produce an interval that is too narrow. We therefore resample *seeds* with replacement and
carry each drawn seed's entire profile across all six resolutions -- a cluster bootstrap.

Both are reported below so the difference is visible rather than asserted.
"""

from __future__ import annotations

import pathlib
import sys
from collections import defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.figdata import load, paired

LOW = (32, 128, 192, 256)          # R <= 256
HIGH = (384, 512)                  # R > 256
N_BOOT = 10_000
SEED = 20260812                    # fixed so the interval is reproducible


def per_seed_outcomes() -> dict[int, dict[int, int]]:
    """{seed_index: {R: 1 if the pair was decided correctly else 0}} for Opus under A1, B=16."""
    cells = defaultdict(list)
    for r in load():
        name, _, val = (r.get("sweep") or "").partition("=")
        if name not in ("resolution", "resolution_fine"):
            continue
        if (r.get("budget") != 16 or r["condition"] != "A1"
                or r["model"] != "claude-opus-5" or r["mechanism"] != "cascade"):
            continue
        # baseline condition only: the robustness cells reuse this sweep name at R=128 with a
        # confounder overlay or at test scale, and pair_id does not encode scale, so they merge in
        if r.get("confounder", "none") != "none" or r.get("scale", "dev") != "dev":
            continue
        cells[int(val)].append(r)

    out: dict[int, dict[int, int]] = defaultdict(dict)
    for R, rows in cells.items():
        _pa, _ev, _n, bits = paired(rows)
        for i, b in enumerate(bits):           # pair ids sort by seed, so index == seed
            out[i][R] = int(b == "1")
    return out


def contrast(seeds: list[int], data: dict[int, dict[int, int]]) -> float:
    """Mean over the drawn seeds of (mean accuracy at low R) - (mean accuracy at high R)."""
    vals = []
    for s in seeds:
        d = data[s]
        lo = np.mean([d[R] for R in LOW if R in d])
        hi = np.mean([d[R] for R in HIGH if R in d])
        vals.append(lo - hi)
    return float(np.mean(vals))


data = per_seed_outcomes()
seeds = sorted(data)
print(f"Opus A1, B=16, cascade. {len(seeds)} seeds x {len(data[seeds[0]])} resolutions.\n")
print(f"  {'seed':<6}" + "".join(f"{R:>7}" for R in LOW + HIGH) + f"{'contrast':>11}")
for s in seeds:
    d = data[s]
    row = "".join(f"{d.get(R, '-'):>7}" for R in LOW + HIGH)
    lo = np.mean([d[R] for R in LOW if R in d])
    hi = np.mean([d[R] for R in HIGH if R in d])
    print(f"  {s:<6}{row}{lo - hi:>11.2f}")

point = contrast(seeds, data)
print(f"\npoint estimate: {point:+.4f}")

rng = np.random.default_rng(SEED)

# --- correct: cluster bootstrap, resampling whole seeds -------------------------------------
cl = np.array([contrast(list(rng.choice(seeds, size=len(seeds), replace=True)), data)
               for _ in range(N_BOOT)])
cl_lo, cl_hi = np.percentile(cl, [2.5, 97.5])

# --- incorrect but instructive: resample (R, seed) cells independently -----------------------
flat_lo = np.array([data[s][R] for s in seeds for R in LOW if R in data[s]], dtype=float)
flat_hi = np.array([data[s][R] for s in seeds for R in HIGH if R in data[s]], dtype=float)
ind = np.array([rng.choice(flat_lo, flat_lo.size, replace=True).mean()
                - rng.choice(flat_hi, flat_hi.size, replace=True).mean()
                for _ in range(N_BOOT)])
ind_lo, ind_hi = np.percentile(ind, [2.5, 97.5])

print(f"\ncluster bootstrap  (resample SEEDS, {N_BOOT:,} draws)   "
      f"{point:+.2f}  [{cl_lo:+.2f}, {cl_hi:+.2f}]   <-- report this")
print(f"naive bootstrap    (resample (R,seed) cells)          "
      f"{point:+.2f}  [{ind_lo:+.2f}, {ind_hi:+.2f}]")
print(f"\nnaive interval is {((cl_hi - cl_lo) / (ind_hi - ind_lo)):.2f}x "
      f"{'wider' if (cl_hi - cl_lo) > (ind_hi - ind_lo) else 'narrower'} than the cluster one; "
      "the cluster interval is the honest one because the same five worlds recur at every R.")
