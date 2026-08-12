"""M3 analysis: per-seed outcomes, and whether A3 decisions follow the operator's own number.

The interesting question on M3 is not the accuracy alone. M3's semantic operator
(`neighborhood_evolution`) differs from M1's in kind: `temporal_reachability` returns a decisive
predicate — the operator effectively answers the question — whereas `neighborhood_evolution`
returns a *measurement* (`peak_change`) that the agent must still judge against a criterion. The
worlds are built so that judgement is easy: positives peak at 1.00, negatives at most 0.29.

So for every A3 run we recover the largest `peak_change` the agent actually saw and compare it
against the decision it made. That separates three very different failures:

  * never called the operator      -> did not acquire the evidence
  * called it, read a low number, said "present" -> had the evidence and misjudged it
  * called it and followed it      -> correct

Only the second is an interpretation failure, and it is the one that bears on whether semantic
abstraction closes the model-capability gap.

    python scripts/analyze_m3.py runs/m3_v2_opus runs/m3_v2_mini
"""

from __future__ import annotations

import json
import pathlib
import sys
from collections import defaultdict

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from worlds.mechanisms.regime_switch import generate_regime_pair

DIRS = [pathlib.Path(a) for a in sys.argv[1:]] or [ROOT / "runs/m3_v2_opus"]

# the parameters the task actually states, recovered from the generator (free, no API calls)
_cfg = yaml.safe_load((ROOT / "configs/mechanisms.yaml").read_text())
focus_by_pair: dict[str, dict] = {}
for _seed in range(5):
    _, _, _spec, _ = generate_regime_pair(json.loads(json.dumps(_cfg)), _seed, scale="dev")
    focus_by_pair[f"m3_none_seed{_seed:03d}"] = {
        "group": list(_spec.group), "window": list(_spec.window), "bandwidth": _spec.bandwidth}


def load(d: pathlib.Path) -> list[dict]:
    seen = {}
    for line in (d / "results.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            seen[r["run_id"]] = r
    return list(seen.values())


for d in DIRS:
    rows = load(d)
    model = rows[0]["model"]
    print(f"\n{'=' * 78}\n{d.name}   model={model}   {len(rows)} runs\n{'=' * 78}")

    # ---------------------------------------------------------- per-cell, per-seed pair outcomes
    cells: dict[tuple, dict] = defaultdict(dict)
    for r in rows:
        cells[(r["condition"], r.get("budget"))].setdefault(r["pair_id"], {})[r["polarity"]] = r
    print(f"{'cond':<5}{'B':>4}{'pairAcc':>9}  per-seed  decisions (pos/neg)")
    for k in sorted(cells, key=lambda x: (x[0], x[1] or 0)):
        pairs = cells[k]
        bits, det = "", []
        for pid in sorted(pairs):
            v = pairs[pid]
            pos = v.get("positive", {}).get("decision", "-")
            neg = v.get("negative", {}).get("decision", "-")
            ok = pos == "present" and neg == "absent"
            bits += "1" if ok else "0"
            det.append(f"{pos[:4]}/{neg[:4]}")
        acc = bits.count("1") / len(bits) if bits else 0.0
        print(f"{k[0]:<5}{k[1] or 0:>4}{acc:>9.2f}  {bits:<8}  {' '.join(det)}")

    # ---------------------------------------------------------- A3: operator number vs decision
    # The reading that bears on the decision is the GOLD-parameterised one -- the call that asks
    # the question the task actually posed. Off-parameter calls (a halved bandwidth, a narrowed
    # window) legitimately return high values on negative worlds, so scoring against the maximum
    # reading would mislabel a correct conclusion drawn in spite of a misleading side-probe as
    # "followed the number". The two columns are therefore kept separate.
    tp = d / "traces"
    print(f"\n  A3 — the operator reading that matched the task's parameters, vs the decision")
    print(f"    {'gold':<6}{'decision':<13}{'gold-param':>11}{'off-param max':>14}  verdict")
    tally = defaultdict(int)
    for r in sorted([x for x in rows if x["condition"] == "A3"],
                    key=lambda x: (x["pair_id"], x["polarity"])):
        t = json.loads((tp / f"{r['run_id']}.json").read_text())
        focus = focus_by_pair.get(r["pair_id"])
        gold_peaks, off_peaks, called = [], [], False
        for s in t["steps"]:
            if s["tool"] != "neighborhood_evolution" or not s.get("ok", True):
                continue
            called = True
            try:
                peak = float(json.loads(s["result_summary"])["peak_change"])
            except (ValueError, TypeError, KeyError):
                continue
            a = s.get("args", {})
            if focus is None:
                off_peaks.append(peak)          # fail safe: unknown reference is not a match
                continue
            matches = (a.get("bandwidth") == focus["bandwidth"]
                       and list(a.get("window") or []) == focus["window"]
                       and a.get("group") is not None
                       and set(map(int, a["group"])) == set(focus["group"]))
            (gold_peaks if matches else off_peaks).append(peak)
        gp = max(gold_peaks) if gold_peaks else None
        op = max(off_peaks) if off_peaks else None
        want = "present" if r["gold"] else "absent"
        correct = r["decision"] == want

        if not called:
            verdict = "NO OPERATOR CALL"
        elif gp is None:
            verdict = ("correct, but never asked with the task's parameters" if correct
                       else "wrong; never asked with the task's parameters")
        elif correct:
            verdict = ("correct despite a misleading off-parameter reading"
                       if op is not None and not r["gold"] and op > 0.30
                       else "followed the gold-parameterised number")
        elif gp <= 0.30 and r["decision"] == "present":
            verdict = "MISREAD low gold-param peak as abrupt"
        elif gp >= 0.90 and r["decision"] != "present":
            verdict = "MISREAD high gold-param peak"
        else:
            verdict = "wrong, other"
        tally[verdict] += 1
        gs = f"{gp:.3f}" if gp is not None else "    --"
        os_ = f"{op:.3f}" if op is not None else "    --"
        print(f"    {str(r['gold']):<6}{r['decision']:<13}{gs:>11}{os_:>14}  {verdict}")
    print("\n    summary:")
    for k, v in sorted(tally.items(), key=lambda x: -x[1]):
        print(f"      {v:>2}  {k}")
