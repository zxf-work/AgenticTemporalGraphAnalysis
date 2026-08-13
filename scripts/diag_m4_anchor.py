"""One-off diagnostic: per-run M4 stage-2 citation windows vs the true t_c."""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from evaluation.error_taxonomy import _compound_stage2_anchored, _compound_stage2_bound
from tasks.lookup import compiled_pair
from traces.schema import Trace

mcfg = yaml.safe_load((ROOT / "configs/mechanisms.yaml").read_text())
xcfg = yaml.safe_load((ROOT / "configs/experiment_matrix.yaml").read_text())
cache = {}

for d in ("m4_opus", "m4_mini"):
    for line in (ROOT / "runs" / d / "results.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["condition"] != "A3":
            continue
        cp = compiled_pair(r, mcfg, xcfg, cache)
        rec = getattr(cp, r["polarity"])
        tc = (cp.compound_tc or {}).get(r["polarity"])
        trace = Trace.from_dict(json.loads(
            (ROOT / "runs" / d / "traces" / f"{r['run_id']}.json").read_text()))
        adm = rec.task.focus.get("admissible")
        claim_pos = (trace.final_claim or {}).get("decision") == "present"
        cited = {e["trace_id"] for e in (trace.final_claim or {}).get("evidence", [])
                 if e.get("role") == "support"}
        rows = []
        for s in trace.steps:
            if s.tool not in ("temporal_reachability", "temporal_paths") or not s.ok:
                continue
            w = s.args.get("window")
            rows.append((s.step_id in cited, w,
                         _compound_stage2_bound(s, rec.task.focus),
                         _compound_stage2_anchored(s, rec.task.focus, tc, claim_pos)))
        print(f"{d} seed={r['seed']} {r['polarity'][:3]} claim={'P' if claim_pos else 'A'} "
              f"t_c={tc} adm={adm}")
        for cited_flag, w, old, new in rows:
            print(f"    {'CITED' if cited_flag else '     '} w={w} old={int(old)} new={int(new)}")
