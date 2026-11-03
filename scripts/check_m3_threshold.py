"""M3 task-text ablation: are the agent's labels explainable by any single decision threshold?

    python scripts/check_m3_threshold.py

Camera-ready check (reviewer 1, C5). In the superseded v2 cells (theta_3 absent from the task
text; `PREREGISTRATION_M.md`, third amendment) gpt-5-mini fell to PairAcc 0.20 while Opus 5
stayed at 1.00. The alternative reading is that gpt-5-mini applied a *different* implicit
threshold to the returned statistic rather than misreading it. That reading predicts the labels
are monotone in the returned peak score: some cut c with every "present" score above c and every
other score at or below it. This script tests that prediction on the stored v2 traces.

Three readings of "the returned score" are tested per run: the gold-parameter call (group,
window and bandwidth equal to the task focus), the maximum over every call, and the maximum over
cited calls. The v2 directories are excluded from the reported corpus by `evaluation/figdata.py`
and are read directly here.
"""

from __future__ import annotations

import json
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from worlds.mechanisms.regime_switch import generate_regime_pair     # noqa: E402

cfg = yaml.safe_load((ROOT / "configs/mechanisms.yaml").read_text())
focus = {}
for seed in range(5):
    _, _, spec, _ = generate_regime_pair(json.loads(json.dumps(cfg)), seed, scale="dev")
    focus[f"m3_none_seed{seed:03d}"] = {"group": set(spec.group), "window": list(spec.window),
                                        "bandwidth": spec.bandwidth}


def monotone(points: list[tuple[float, str]]) -> tuple[bool, tuple[float, float] | None]:
    """Is there a cut c with every 'present' score > c >= every other score?"""
    pos = [s for s, lab in points if lab == "present"]
    neg = [s for s, lab in points if lab != "present"]
    if not pos or not neg:
        return True, None
    return min(pos) > max(neg), (max(neg), min(pos))


for d in ("m3_v2_mini", "m3_v2_opus"):
    src = ROOT / "runs" / d / "results.jsonl"
    rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["condition"] == "A3"]
    print(f"\n=== {d}: A3, B=16, theta_3 unstated ===")
    pts = {"gold-parameter call": [], "max over all calls": [], "max over cited calls": []}
    for r in sorted(rows, key=lambda x: (x["pair_id"], x["polarity"])):
        t = json.loads((ROOT / "runs" / d / "traces" / f"{r['run_id']}.json").read_text())
        f = focus[r["pair_id"]]
        cited = {e["trace_id"] for e in (t.get("final_claim") or {}).get("evidence", [])
                 if e.get("role") == "support"}
        gold_s, all_s, cited_s = [], [], []
        for s in t["steps"]:
            if s["tool"] != "neighborhood_evolution" or not s.get("ok", True):
                continue
            try:
                peak = json.loads(s["result_summary"])["peak_change"]
            except (ValueError, KeyError, TypeError):
                continue
            a = s["args"]
            is_gold = (a.get("bandwidth") == f["bandwidth"]
                       and list(a.get("window") or []) == f["window"]
                       and a.get("group") is not None and set(map(int, a["group"])) == f["group"])
            all_s.append(peak)
            if is_gold:
                gold_s.append(peak)
            if s["step_id"] in cited:
                cited_s.append(peak)
        lab = r["decision"]
        gold = "present" if r["gold"] else "absent"
        print(f"  {r['pair_id']} {r['polarity'][:3]} gold={gold:<7} label={lab:<12} "
              f"gold-param peak={max(gold_s) if gold_s else None}  all={sorted(all_s)}")
        for name, vals in (("gold-parameter call", gold_s), ("max over all calls", all_s),
                           ("max over cited calls", cited_s)):
            if vals:
                pts[name].append((max(vals), lab))
    for name, p in pts.items():
        ok, gap = monotone(p)
        print(f"  [{name}] n={len(p)} sorted={sorted((round(s, 3), lab[:4]) for s, lab in p)}")
        print(f"     a single threshold reproduces the labels: {ok}"
              + (f"  (max non-present {gap[0]:.3f}, min present {gap[1]:.3f})" if gap else ""))
