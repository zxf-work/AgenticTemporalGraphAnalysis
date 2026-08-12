"""Run a predeclared sweep from configs/experiment_matrix.yaml.

    python scripts/run_matrix.py <sweep> [--out DIR] [--conc N] [--seeds N] [--dry]
    python scripts/run_matrix.py ambiguity --seeds 2 --dry     # plan only, no API calls

Sweeps are declared in the config and frozen before running; this script only selects which one to
execute. Everything is resumable: re-running the same sweep reuses completed episodes, so an
interrupted run costs only what it had not finished.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.backends import load_subject, warmup
from agents.runner import Cell, MatrixRunner, load_results
from evaluation.metrics import summarize

ap = argparse.ArgumentParser()
ap.add_argument("sweep", help="name of a sweep in configs/experiment_matrix.yaml")
ap.add_argument("--out", default=None, help="output dir (default runs/<sweep>)")
ap.add_argument("--conc", type=int, default=None, help="concurrency")
ap.add_argument("--seeds", type=int, default=None, help="pairs per cell")
ap.add_argument("--budgets", default=None, help="comma-separated override")
ap.add_argument("--conditions", default=None, help="comma-separated override")
ap.add_argument("--models", default=None, help="comma-separated model slots")
ap.add_argument("--dry", action="store_true", help="print the plan and exit")
ap.add_argument("--values", default=None,
                help="comma-separated subset of the sweep's axis values")
ap.add_argument("--budget", type=float, default=None,
                help="hard USD ceiling on Anthropic spend; sweep stops when reached")
ap.add_argument("--confounder", default=None,
                help="confounder overlay applied identically to both worlds, e.g. "
                     "c1_global_burst_high (default: the sweep's declared confounder)")
ap.add_argument("--scale", default=None, choices=["dev", "test"],
                help="world scale; test uses background.n_nodes_test")
args = ap.parse_args()

mcfg = yaml.safe_load((ROOT / "configs/mechanisms.yaml").read_text())
xcfg = yaml.safe_load((ROOT / "configs/experiment_matrix.yaml").read_text())
models_cfg = yaml.safe_load((ROOT / "configs/models.yaml").read_text())

if args.sweep not in xcfg["sweeps"]:
    sys.exit(f"unknown sweep {args.sweep!r}; available: {list(xcfg['sweeps'])}")
spec = xcfg["sweeps"][args.sweep]
d = xcfg["defaults"]

n_seeds = args.seeds or d["pairs_per_cell"]
budgets = ([int(b) for b in args.budgets.split(",")] if args.budgets
           else spec.get("budgets", xcfg["budgets"]))
conditions = (args.conditions.split(",") if args.conditions else xcfg["conditions"])
model_slots = (args.models.split(",") if args.models
               else xcfg["models"]["llm"] + xcfg["models"]["reference"])

# ---------------------------------------------------------------- build cells
axis = spec["axis"]
cells: list[Cell] = []
axis_values = spec["values"]
if args.values:
    want = args.values.split(",")
    axis_values = [v for v in axis_values if str(v) in want]
    if not axis_values:
        sys.exit(f"--values {args.values} matches none of {spec['values']}")
for value in axis_values:
    mechanisms = [value] if axis == "mechanism" else spec.get("mechanisms", ["cascade"])
    overrides = tuple(sorted(spec.get("hold", {}).items()))
    if axis != "mechanism":
        overrides = tuple(sorted({**dict(overrides), axis: value}.items()))
    for mech in mechanisms:
        for seed in range(n_seeds):
            for cond in conditions:
                for slot in model_slots:
                    if slot == "script_a1" and cond != "A1":
                        continue          # the reference algorithm only inhabits the A1 interface
                    for b in budgets:
                        cells.append(Cell(
                            mechanism=mech, seed=seed,
                            confounder=args.confounder or d["confounders"][0],
                            condition=cond, model_slot=slot, budget=b,
                            paraphrase=d["paraphrase"], scale=args.scale or d["scale"],
                            sweep=f"{args.sweep}={value}", overrides=overrides))

out = pathlib.Path(args.out) if args.out else (ROOT / "runs" / args.sweep)
conc = args.conc or d["concurrency"]

print(f"sweep '{args.sweep}' on axis '{axis}' over {axis_values}")
print(f"  conditions={conditions} models={model_slots} budgets={budgets} seeds={n_seeds}")
print(f"  {len(cells)} cells -> {2 * len(cells)} runs, concurrency={conc}")
print(f"  output: {out}")
llm_runs = 2 * sum(1 for c in cells if c.model_slot != "script_a1")
print(f"  LLM episodes: {llm_runs} (~{llm_runs * 90 / conc / 3600:.1f}h at 90s each)")
from evaluation.cost import estimate_sweep
for slot in model_slots:
    if slot == "script_a1":
        continue
    mid = models_cfg["models"][slot]["id"]
    n = 2 * sum(1 for c in cells if c.model_slot == slot)
    print(f"    {slot:10s} {mid:18s} {n:5d} episodes  ~${estimate_sweep(n, mid):.2f}")
if args.budget:
    print(f"  hard ceiling: ${args.budget:.2f} on Anthropic spend")
if args.dry:
    for c in cells[:6]:
        print("   ", c.key())
    print(f"    ... ({len(cells)} cells total)")
    sys.exit(0)

# ---------------------------------------------------------------- backends
backends: dict[str, object] = {}
for slot in model_slots:
    if slot == "script_a1":
        backends[slot] = type("ScriptRef", (), {"name": "script-a1-bisect"})()
        continue
    b = load_subject(models_cfg, slot)
    status = warmup(b)                     # settle accepted request shape before fanning out
    print(f"  backend {slot}: {b.name} ({status})")
    if status != "ok":
        sys.exit(f"backend {slot} unusable: {status}")
    backends[slot] = b

runner = MatrixRunner(out, backends=backends, concurrency=conc, mech_cfg=mcfg,
                      budget_usd=args.budget)
runner.write_manifest(cells, extra={"sweep": args.sweep, "axis": axis,
                                    "values": spec["values"], "budgets": budgets,
                                    "conditions": conditions, "n_seeds": n_seeds})
t0 = time.time()
rows = runner.run(cells)
runner.close()
print(f"\ncompleted in {(time.time() - t0) / 60:.1f} min; {len(rows)} rows")
print(runner.budget.line())

# ---------------------------------------------------------------- summary
recs = load_results(out / "results.jsonl")
print(f"\n{'sweep':<22}{'cond':<6}{'B':>4}{'pairAcc':>9}{'steps':>7}{'inTok':>8}"
      f"{'paramErr':>9}{'recon':>7}")
groups: dict[tuple, list] = {}
for r in recs:
    groups.setdefault((r.get("sweep", ""), r["condition"], r.get("budget")), []).append(r)
for k in sorted(groups, key=lambda x: (str(x[0]), str(x[1]), x[2] or 0)):
    g = groups[k]
    pairs_ok = {}
    for r in g:
        pairs_ok.setdefault(r["pair_id"], {})[r["polarity"]] = r["decision"]
    acc = sum(1 for v in pairs_ok.values()
              if v.get("positive") == "present" and v.get("negative") == "absent")
    n = len(pairs_ok)
    print(f"{str(k[0]):<22}{k[1]:<6}{k[2] or 0:>4}{(acc / n if n else 0):>9.2f}"
          f"{sum(r['tool_calls'] for r in g) / len(g):>7.1f}"
          f"{sum(r['input_tokens'] for r in g) / len(g):>8.0f}"
          f"{sum(r['param_error_rate'] for r in g) / len(g):>9.2f}"
          f"{sum(bool(r['used_reconstruction']) for r in g) / len(g):>7.2f}")
print(f"\nresults: {out / 'results.jsonl'}   manifest: {out / 'manifest.json'}")
