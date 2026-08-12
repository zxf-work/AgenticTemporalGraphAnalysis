"""Script-A1-Bisect reference measurement (T16).

Reports calls-to-decision and accuracy for the deterministic reconstruction algorithm at several
budgets, giving the algorithmic reference point the LLM curves are compared against.

Run:  python scripts/test_script_a1.py [n_pairs] [budget,...]
"""
from __future__ import annotations
import pathlib, sys
import numpy as np, yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.script_a1 import run_script_a1
from tasks.compiler import compile_pair, load_world

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
BUDGETS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else "20,40,80,200".split(","))]
cfg = yaml.safe_load((ROOT / "configs/mechanisms.yaml").read_text())

pairs = [compile_pair(cfg, "cascade", s, scale="dev") for s in range(N)]
print(f"Script-A1-Bisect on {N} cascade pairs "
      f"(decoys={cfg['m1_cascade'].get('n_decoy_paths', 0)}, "
      f"max_wait={cfg['m1_cascade']['max_wait']})\n")
print(f"{'budget':>7} {'pairAcc':>8} {'1world':>7} {'meanCalls':>10} {'medCalls':>9} "
      f"{'exhausted':>10} {'meanCands':>10}")

for B in BUDGETS:
    correct_pairs = single = 0
    calls, exhausted, cands = [], 0, []
    for cp in pairs:
        got = {}
        for rec in (cp.positive, cp.negative):
            w = load_world(rec.world_path)
            r = run_script_a1(w, rec.task, tool_budget=B, world_hash=rec.world_hash,
                              pair_id=rec.pair_id)
            got[rec.polarity] = r.decision
            calls.append(r.calls)
            cands.append(r.candidates)
            exhausted += (r.decision == "insufficient")
            single += (r.decision == ("present" if rec.gold else "absent"))
        correct_pairs += (got["positive"] == "present" and got["negative"] == "absent")
    n = 2 * len(pairs)
    print(f"{B:7d} {correct_pairs/len(pairs):8.2f} {single/n:7.2f} {np.mean(calls):10.1f} "
          f"{np.median(calls):9.1f} {exhausted/n:10.2f} {np.mean(cands):10.1f}")

print("\nPrediction: calls ~ (candidate routes) x (route length) x log_W(span/(max_wait/2)).")
