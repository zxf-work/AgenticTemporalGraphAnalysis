"""Shared data extraction for the paper figures.

Figures read from `runs/*/results.jsonl` rather than transcribed tables, so a figure cannot drift
away from the evidence behind it. Building Figure 1 this way is how the per-seed transcription
error in `PAPER_CLAIMS.md` was caught.
"""

from __future__ import annotations

import json
import pathlib
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"

MODELS = {"claude-opus-5": "Opus 5", "gpt-5-mini": "gpt-5-mini"}
#: A1 dark, A2 orange, A3 blue — colourblind-safe
C = {"A1": "#222222", "A2": "#D55E00", "A3": "#0072B2", "mini": "#8C8C8C"}

#: superseded by a preregistered amendment, or invalidated by the world-collision bug.
#: m3_v2_* are the second-amendment M3 runs, superseded by the third amendment (theta_3 stated in
#: the task text; PREREGISTRATION_M.md) whose runs are m3_v3_*.
SKIP_DIRS = ("m3_mini", "m3_opus", "m3_v2_mini", "m3_v2_opus")


def _superseded(d: str, r: dict) -> bool:
    """Rows that are stale for a reason the directory name alone does not carry.

    The cross-mechanism dirs hold BOTH mechanisms, but only their regime_switch half was
    invalidated by the M3 margin amendment (`PREREGISTRATION_M.md`); the closure_wave half stands.
    Dropping the whole directory would throw away valid M2 data, and keeping it silently mixed
    pre- and post-amendment M3 worlds in the same average. `replay_backend.py` independently
    confirms these rows are stale: their world hashes no longer recompile.
    """
    return d.startswith("xmech_") and r.get("mechanism") == "regime_switch"


def load() -> list[dict]:
    rows, seen = [], set()
    for f in sorted((ROOT / "runs").glob("*/results.jsonl")):
        d = f.parent.name
        if "INVALID" in d or "PRE_AMENDMENT" in d or d in SKIP_DIRS:
            continue
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            key = (d, r.get("run_id"))
            if key in seen:
                continue
            seen.add(key)
            r["_dir"] = d
            if _superseded(d, r):
                continue
            rows.append(r)
    return rows


def ev_valid(r: dict) -> bool:
    """EV_i: the answer is right AND the cited computation preserved the task's semantics.

    Defined *existentially* over cited support — see `evaluation/error_taxonomy.py`. A run is
    evidence-valid when at least one cited computation establishes the claim under the task's
    constraints; a broad reconnaissance call cited alongside a correctly bound one does not
    disqualify it.
    """
    return bool((r.get("check") or {}).get("evidence_grounded")) and not r.get("cited_bad_support")


def paired(rows: list[dict]) -> tuple[float, float, int, str]:
    """(PairAcc, EVPairAcc, n_pairs, per-seed bit string) for one cell."""
    by = defaultdict(dict)
    for r in rows:
        by[r["pair_id"]][r["polarity"]] = r
    complete = [by[k] for k in sorted(by) if len(by[k]) == 2]
    if not complete:
        return 0.0, 0.0, 0, ""
    bits, ok, ev = "", 0, 0
    for v in complete:
        good = v["positive"]["decision"] == "present" and v["negative"]["decision"] == "absent"
        bits += "1" if good else "0"
        ok += good
        ev += good and ev_valid(v["positive"]) and ev_valid(v["negative"])
    return ok / len(complete), ev / len(complete), len(complete), bits


def by_interface(rows: list[dict], mechanism: str) -> dict[tuple, tuple]:
    """{(condition, model): (PairAcc, EVPairAcc, n, bits)} for one mechanism."""
    g = defaultdict(list)
    for r in rows:
        if r.get("mechanism") == mechanism and r.get("model") in MODELS:
            g[(r["condition"], r["model"])].append(r)
    return {k: paired(v) for k, v in g.items()}


def resolution_axis(rows: list[dict]) -> dict[tuple, dict[int, tuple]]:
    """{(condition, model): {R: (PairAcc, EVPairAcc, n, bits)}} for the M1 cascade R sweep.

    A1/A3 come from the `resolution` sweeps and A2 from `a2_m1`; all three are declared with
    identical hold values (hops=5, D=0, B=16), so they are the same cells.
    """
    out = defaultdict(dict)
    for r in rows:
        name, _, val = (r.get("sweep") or "").partition("=")
        if name not in ("resolution", "resolution_fine", "a2_m1") or r.get("budget") != 16:
            continue
        if r.get("mechanism") != "cascade" or r.get("model") not in MODELS:
            continue
        # The robustness cells reuse the `resolution` sweep name at R=128 with a confounder overlay
        # or at test scale, and `pair_id` encodes mechanism/confounder/seed/variant but NOT scale --
        # so the 8K worlds collide with the dev ones and silently merged into this curve, turning
        # R=128 from 5 pairs into 10 and changing its value. The main curve is the baseline
        # condition only; the robustness cells are reported separately.
        if r.get("confounder", "none") != "none" or r.get("scale", "dev") != "dev":
            continue
        try:
            R = int(val)
        except ValueError:
            continue
        out[(r["condition"], r["model"])].setdefault(R, []).append(r)
    return {k: {R: paired(v) for R, v in d.items()} for k, d in out.items()}


def rc_params() -> dict:
    return {
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 200, "savefig.bbox": "tight", "pdf.fonttype": 42,
    }
