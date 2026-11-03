"""M4 audit-boundary sensitivity: would the *returned payloads* decide the question?

    python scripts/diag_m4_payload.py

Camera-ready diagnostic (reviewers 1, 3, 4). The reported M4 criterion asks whether a cited
stage-2 query is *anchored* at the oracle closure time t_c (`error_taxonomy._compound_stage2_anchored`).
The excluded inference -- comparing returned arrival times against [t_c, t_c + admissible] -- is
itself deterministic, so this script recomputes it over the cited reachability / paths payloads of
every M4 A3 episode and reports how many matched pairs would count as evidence-valid under a
"cited payload plus deterministic interval comparison" variant. It is a sensitivity analysis of
where the audit boundary sits, not a replacement metric: it accepts a composition the agent
performed in prose, which the reported criterion deliberately does not.

Rule per cited payload, with I = [t_c, t_c + admissible]:
  * supports *present* iff some returned route has every timestamp inside I;
  * supports *absent* iff the query window covers I and no returned route lies inside I, and
    either the call returned reached=false / no paths, or (temporal_reachability) the
    earliest-arrival route arrives after I;
  * a `temporal_paths` call whose returned routes all lie outside I is inconclusive under the
    strict reading (the enumeration is bounded by max_paths); the "exhaustive" reading treats a
    call that returned fewer than max_paths routes as a complete enumeration.
"""

from __future__ import annotations

import json
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tasks.lookup import compiled_pair      # noqa: E402
from traces.schema import Trace             # noqa: E402

mcfg = yaml.safe_load((ROOT / "configs/mechanisms.yaml").read_text())
xcfg = yaml.safe_load((ROOT / "configs/experiment_matrix.yaml").read_text())
cache: dict = {}
DEFAULT_MAX_PATHS = 3


def routes_of(step) -> tuple[bool | None, list[list[int]]]:
    try:
        res = json.loads(step.result_summary)
    except (ValueError, TypeError):
        return None, []
    if step.tool == "temporal_reachability":
        return bool(res.get("reached")), ([[h["t"] for h in res.get("route", [])]]
                                          if res.get("reached") else [])
    if step.tool == "temporal_paths":
        paths = res.get("paths") or []
        return bool(paths), [[e[2] for e in p] for p in paths]
    return None, []


def verdict(step, tc: int, adm: int, exhaustive: bool) -> str:
    lo, hi = tc, tc + adm
    reached, routes = routes_of(step)
    w = step.args.get("window")
    covers = isinstance(w, list) and len(w) == 2 and int(w[0]) <= lo and int(w[1]) >= hi
    if reached is None:
        return "inconclusive"
    if reached and any(all(lo <= t <= hi for t in r) for r in routes):
        return "present"
    if not reached:
        return "absent" if covers else "inconclusive"
    # reached, but no route inside the interval
    if not covers:
        return "inconclusive"
    if step.tool == "temporal_reachability" and min(r[-1] for r in routes) > hi:
        return "absent"
    if step.tool == "temporal_paths" and exhaustive:
        if len(routes) < int(step.args.get("max_paths") or DEFAULT_MAX_PATHS):
            return "absent"
    return "inconclusive"


summary = {}
for d in ("m4_mini", "m4_opus"):
    rows = [json.loads(l) for l in (ROOT / "runs" / d / "results.jsonl").read_text().splitlines()
            if l.strip()]
    rows = [r for r in rows if r["condition"] == "A3"]
    print(f"\n=== {d}: A3, B=32 ===")
    per_pair: dict[int, dict[str, dict[str, bool]]] = {}
    for r in sorted(rows, key=lambda x: (x["seed"], x["polarity"])):
        cp = compiled_pair(r, mcfg, xcfg, cache)
        rec = getattr(cp, r["polarity"])
        tc = cp.compound_tc[r["polarity"]]
        adm = int(rec.task.focus["admissible"])
        tr = Trace.from_dict(json.loads((ROOT / "runs" / d / "traces" / f"{r['run_id']}.json")
                                        .read_text()))
        assert tr.world_hash == rec.world_hash
        gold = "present" if r["gold"] else "absent"
        cited = {e["trace_id"] for e in (tr.final_claim or {}).get("evidence", [])
                 if e.get("role") == "support"}
        official = bool(r["check"]["evidence_grounded"]) and not r.get("cited_bad_support")
        out = {}
        for mode in ("strict", "exhaustive"):
            vs = [verdict(s, tc, adm, mode == "exhaustive") for s in tr.steps
                  if s.ok and s.step_id in cited
                  and s.tool in ("temporal_reachability", "temporal_paths")]
            out[mode] = r["decision"] == gold and gold in vs
        per_pair.setdefault(r["seed"], {})[r["polarity"]] = out
        cited_desc = [(s.step_id, s.tool, s.args.get("window"), routes_of(s)[1])
                      for s in tr.steps if s.ok and s.step_id in cited
                      and s.tool in ("temporal_reachability", "temporal_paths")]
        print(f"  seed {r['seed']} {r['polarity'][:3]} gold={gold:<7} label={r['decision']:<8} "
              f"I=[{tc},{tc + adm}] anchored-rule EV={official!s:<5} payload EV strict={out['strict']!s:<5} "
              f"exhaustive={out['exhaustive']!s:<5} cited={cited_desc}")
    res = {}
    for mode in ("strict", "exhaustive"):
        res[mode] = sum(all(v[p][mode] for p in ("positive", "negative")) for v in per_pair.values())
    runs = {mode: sum(v[p][mode] for v in per_pair.values() for p in ("positive", "negative"))
            for mode in ("strict", "exhaustive")}
    summary[d] = (res, runs, len(per_pair))
    print(f"  evidence-valid PAIRS under cited-payload + interval comparison: "
          f"strict {res['strict']}/{len(per_pair)}, exhaustive-paths {res['exhaustive']}/{len(per_pair)}"
          f"  (runs: strict {runs['strict']}/{2 * len(per_pair)}, exhaustive {runs['exhaustive']}/{2 * len(per_pair)})")

print("\nSUMMARY (pairs evidence-valid; reported anchored rule in parentheses)")
for d, (res, runs, n) in summary.items():
    print(f"  {d}: strict {res['strict']}/{n}, exhaustive {res['exhaustive']}/{n}")
