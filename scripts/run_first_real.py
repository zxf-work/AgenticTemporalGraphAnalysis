"""First real agent runs: one matched cascade pair, conditions A1 and A3, live model.

Purpose is to prove the wire format and the loop against a real provider before spending a matrix
on it. Prints the full trajectory so the tool calls and the reasoning can be inspected by hand,
which is the Aug 13 dev-inspection requirement (T2.5) in miniature.

Run:  python scripts/run_first_real.py [primary|secondary] [n_pairs] [mechanism]
"""

from __future__ import annotations

import json
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.backends import load_subject
from agents.harness import run_episode
from evaluation.metrics import summarize
from evaluation.score_run import score
from tasks.compiler import compile_pair, load_world

WHICH = sys.argv[1] if len(sys.argv) > 1 else "primary"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 1
MECH = sys.argv[3] if len(sys.argv) > 3 else "cascade"
CONDS = (sys.argv[4].split(",") if len(sys.argv) > 4 else ["A0", "A1", "A3"])

mcfg = yaml.safe_load((ROOT / "configs/mechanisms.yaml").read_text())
models = yaml.safe_load((ROOT / "configs/models.yaml").read_text())
backend = load_subject(models, WHICH)
budget = models["defaults"]["tool_budget"]

print(f"model: {backend.name}   mechanism: {MECH}   pairs: {N}   tool budget: {budget}\n")

out = ROOT / "runs" / "first_real"
out.mkdir(parents=True, exist_ok=True)

rows = []
for seed in range(N):
    cp = compile_pair(mcfg, MECH, seed, scale="dev")
    for cond in CONDS:
        for rec in (cp.positive, cp.negative):
            world = load_world(rec.world_path)
            tr = run_episode(task=rec.task, world=world, condition=cond, backend=backend,
                             pair_id=rec.pair_id, world_hash=rec.world_hash,
                             tool_budget=budget)
            row = score(tr, rec, cond)
            rows.append(row)
            (out / f"{tr.run_id}.json").write_text(json.dumps(tr.to_dict(), indent=1))

            gold_s = "PRESENT" if rec.gold else "ABSENT "
            mark = "OK " if row.check.decision_correct else "XX "
            print(f"{mark} {cond} {rec.polarity:8s} gold={gold_s} got={row.decision:12s} "
                  f"steps={row.tool_calls:2d} in={row.input_tokens:6d} "
                  f"out={row.output_tokens:5d} {row.latency_ms/1000:5.1f}s "
                  f"[{row.status}] {row.failure_category}")
            if row.n_temporal_calls:
                print(f"      temporal calls={row.n_temporal_calls} "
                      f"param_err={row.param_error_rate:.2f} "
                      f"cited_bad_support={row.cited_bad_support}")
            if row.range_calls:
                print(f"      range search: calls={row.range_calls} "
                      f"windows/call={row.windows_per_call:.1f} "
                      f"min_width={row.min_window_width} "
                      f"reconstruction={row.used_reconstruction}")
            if tr.final_claim:
                print(f"      says : {tr.final_claim.get('narrative', '')[:200]}")
            if tr.error:
                print(f"      ERROR: {tr.error[:300]}")
            if row.check.reasons:
                print(f"      notes: {row.check.reasons}")
            print()

print("=" * 78)
hdr = (f"{'cond':5s} {'pairAcc':>7s} {'1world':>7s} {'evGrnd':>7s} {'steps':>6s} "
       f"{'inTok':>7s} {'lat_s':>6s} {'paramErr':>8s} {'badSup':>7s} {'recon':>6s}")
print(hdr)
for cond in CONDS:
    s = summarize([r for r in rows if r.condition == cond])
    print(f"{cond:5s} {s.pair_acc:7.2f} {s.single_world_acc:7.2f} "
          f"{s.evidence_grounded_acc:7.2f} {s.mean_tool_calls:6.1f} "
          f"{s.mean_input_tokens:7.0f} {s.mean_latency_s:6.1f} "
          f"{s.param_error_rate:8.2f} {s.cited_bad_support_rate:7.2f} "
          f"{s.reconstruction_rate:6.2f}")
    if s.failure_mix:
        print(f"      failures: {s.failure_mix}")
print(f"\ntraces written to {out}")
