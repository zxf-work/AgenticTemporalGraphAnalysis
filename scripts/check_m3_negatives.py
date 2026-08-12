"""Validity check: do HIGH peak_change readings on negative worlds come from off-parameter calls?

The M3 worlds are built so a negative scores at most `threshold - required_margin` = 0.30 under the
*gold* parameters (the focal group, the stated window, bandwidth 5000). Yet the A3 traces show
`peak_change` up to 1.000 on negative worlds.

Two very different explanations, which must not be conflated:

  * the agent queried with parameters other than the ones the task states (a different bandwidth,
    a wider window, a different group) and found a high reading elsewhere -- the worlds are fine,
    and this is a parameterisation observation, i.e. Finding 3; or
  * a gold-parameterised call returns a high reading on a negative -- the worlds are broken and
    every M3 number is void.

Two independent pieces of evidence are printed. First the gold statistic recomputed straight from
the generator, which does not involve the agent at all and is the decisive test. Then, for each
high reading in a trace, whether the call that produced it used the gold parameters.

NOTE ON FAIL-SAFE DEFAULTS. The first version of this script looked up the task focus in a file
keyed by a `task_id` field that `results.jsonl` does not carry, so the focus was always empty and
every `focus.get(x) is None` comparison defaulted to "matches gold" -- reporting all six readings
as GOLD PARAMS and declaring M3 void. That is the same vacuous-truth bug that once made three
protocol-gate checks pass on a missing directory. A comparison against an unknown reference must
count as "cannot tell", never as "matches".
"""

from __future__ import annotations

import json
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tasks.compiler import compile_pair
from worlds.mechanisms.regime_switch import generate_regime_pair, regime_gold

RUN = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "runs/m3_v2_opus")
cfg = yaml.safe_load((ROOT / "configs/mechanisms.yaml").read_text())
m3 = cfg["m3_regime_switch"]
THRESH, MARGIN = m3["threshold"], m3["required_margin"]

# ------------------------------------------------------- 1. agent-independent gold recomputation
print(f"gold statistic recomputed from the generator (threshold={THRESH}, margin={MARGIN})")
print(f"  required: positive >= {THRESH + MARGIN:.2f}, negative <= {THRESH - MARGIN:.2f}")
focus_by_pair: dict[str, dict] = {}
worlds_ok = True
for seed in range(5):
    c = json.loads(json.dumps(cfg))
    pos_w, neg_w, spec, _ = generate_regime_pair(c, seed, scale="dev")
    kw = dict(group=spec.group, window=spec.window, bandwidth=spec.bandwidth,
              threshold=THRESH)
    ps = regime_gold(pos_w, **kw)[1]
    ns = regime_gold(neg_w, **kw)[1]
    ok = ps >= THRESH + MARGIN and ns <= THRESH - MARGIN
    worlds_ok &= ok
    print(f"  seed{seed}: positive {ps:.3f}   negative {ns:.3f}   {'ok' if ok else 'VIOLATION'}")
    focus_by_pair[f"m3_none_seed{seed:03d}"] = {
        "group": list(spec.group), "window": list(spec.window), "bandwidth": spec.bandwidth}

print(f"\n  -> negative worlds are {'CLEAN' if worlds_ok else 'NOT CLEAN'} under gold parameters")

# ------------------------------------------------------- 2. what parameters did the agent use?
rows = {}
for line in (RUN / "results.jsonl").read_text().splitlines():
    if line.strip():
        r = json.loads(line)
        rows[r["run_id"]] = r

print(f"\nhigh (>0.30) peak_change readings on NEGATIVE worlds in {RUN.name}:")
print(f"  {'pair':<20}{'peak':>7}  {'verdict':<14} how the call differed from the task's parameters")
n_gold_high = n_off = 0
for r in sorted(rows.values(), key=lambda x: x["pair_id"]):
    if r["condition"] != "A3" or r["polarity"] != "negative":
        continue
    t = json.loads((RUN / "traces" / f"{r['run_id']}.json").read_text())
    focus = focus_by_pair.get(r["pair_id"].split("_")[0] + "_none_" + r["pair_id"].split("_")[-1]) \
        or focus_by_pair.get(r["pair_id"])
    for s in t["steps"]:
        if s["tool"] != "neighborhood_evolution" or not s.get("ok", True):
            continue
        try:
            peak = float(json.loads(s["result_summary"])["peak_change"])
        except (ValueError, TypeError, KeyError):
            continue
        if peak <= 0.30:
            continue
        a = s.get("args", {})
        if focus is None:                      # fail safe: unknown reference is NOT a match
            print(f"  {r['pair_id']:<20}{peak:>7.3f}  {'CANNOT TELL':<14} no focus recovered")
            continue
        why = []
        if a.get("bandwidth") != focus["bandwidth"]:
            why.append(f"bandwidth {a.get('bandwidth')} vs {focus['bandwidth']}")
        if list(a.get("window") or []) != focus["window"]:
            why.append(f"window {a.get('window')} vs {focus['window']}")
        if a.get("group") is None or set(map(int, a["group"])) != set(focus["group"]):
            why.append("group differs")
        gold_params = not why
        n_gold_high += int(gold_params)
        n_off += int(not gold_params)
        tag = "GOLD PARAMS" if gold_params else "off-parameter"
        print(f"  {r['pair_id']:<20}{peak:>7.3f}  {tag:<14} {'; '.join(why)}")

print(f"\n{n_off} high readings came from off-parameter calls (expected; Finding 3).")
print(f"{n_gold_high} came from GOLD-parameterised calls.")
print("M3 worlds valid: 0 gold-parameterised high readings on negatives." if n_gold_high == 0
      else "NON-ZERO gold-parameterised high readings -- M3 would be void.")
