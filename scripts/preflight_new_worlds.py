"""Prove that a re-run will actually re-run, before paying for it.

Written after a config fix was committed but never synced: the "corrected" M3 re-run regenerated
byte-identical worlds, produced run ids identical to the ones it was meant to supersede, and cost
$11.59 for a duplicate of a superseded configuration. Two faults compounded --

  1. the remote config was stale, so the worlds did not change; and
  2. the output directory was new, so the resume cache (which keys on trace files *inside the
     output dir*) could not recognise the duplicates and skip them.

Neither fault is visible from the runner's output: it reports `cached=False` and happily bills.
The only cheap, decisive signal is the run id itself, which is a hash over
(task_id, condition, model, prompt, world_hash). If a config change was meant to change the
worlds, the run ids MUST change. If they do not, the change did not take effect.

This script computes the run ids a sweep *would* produce and diffs them against every run id
already on disk. World compilation is free (no API calls), so this costs nothing to run and is
the last gate before spending.

    python scripts/preflight_new_worlds.py cross_mechanism --values regime_switch \
        --conditions A1,A3 --budgets 8,32,16 --models primary,secondary --expect new

`--expect new`  : refuse (exit 1) if ANY run id already exists  -- for a corrected re-run.
`--expect cached`: refuse if any run id is missing              -- for a resume you expect free.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.harness import system_prompt
from common import digest_text, stable_hash
from tasks.compiler import compile_pair
from traces.schema import make_run_id

ap = argparse.ArgumentParser()
ap.add_argument("sweep")
ap.add_argument("--values", default=None)
ap.add_argument("--conditions", default=None)
ap.add_argument("--budgets", default=None)
ap.add_argument("--models", default=None)
ap.add_argument("--seeds", type=int, default=None)
ap.add_argument("--expect", choices=["new", "cached", "report"], default="report")
args = ap.parse_args()

mcfg = yaml.safe_load((ROOT / "configs/mechanisms.yaml").read_text())
xcfg = yaml.safe_load((ROOT / "configs/experiment_matrix.yaml").read_text())
models_cfg = yaml.safe_load((ROOT / "configs/models.yaml").read_text())

spec = xcfg["sweeps"][args.sweep]
d = xcfg["defaults"]
axis = spec["axis"]
n_seeds = args.seeds or d["pairs_per_cell"]
budgets = [int(b) for b in args.budgets.split(",")] if args.budgets else spec.get(
    "budgets", xcfg["budgets"])
conditions = args.conditions.split(",") if args.conditions else xcfg["conditions"]
slots = args.models.split(",") if args.models else xcfg["models"]["llm"]

values = spec["values"]
if args.values:
    want = args.values.split(",")
    values = [v for v in values if str(v) in want]

print(f"preflight: sweep={args.sweep} values={values} conditions={conditions} "
      f"budgets={budgets} models={slots} seeds={n_seeds}")
print(f"mechanism config hash: {stable_hash(mcfg, length=12)}")
for mk in ("m1_cascade", "m2_closure_wave", "m3_regime_switch"):
    print(f"  {mk}: threshold={mcfg[mk]['threshold']} "
          f"required_margin={mcfg[mk]['required_margin']}")

# ------------------------------------------------------------------ every run id already on disk
existing: dict[str, str] = {}
for rf in sorted((ROOT / "runs").glob("*/results.jsonl")):
    for line in rf.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            existing.setdefault(r.get("run_id", ""), rf.parent.name)
print(f"{len(existing)} run ids already on disk across "
      f"{len({v for v in existing.values()})} run dirs")

# ------------------------------------------------------------------ what this sweep would produce
planned: list[tuple[str, str]] = []
infeasible: list[str] = []
for value in values:
    mechanisms = [value] if axis == "mechanism" else spec.get("mechanisms", ["cascade"])
    overrides = tuple(sorted(spec.get("hold", {}).items()))
    if axis != "mechanism":
        overrides = tuple(sorted({**dict(overrides), axis: value}.items()))
    variant = stable_hash(sorted(overrides), length=6) if overrides else ""
    for mech in mechanisms:
        for seed in range(n_seeds):
            cfg = json.loads(json.dumps(mcfg))       # deep copy per cell, as the runner does
            try:
                cp = compile_pair(cfg, mech, seed, confounder=d["confounders"][0],
                                  scale=d["scale"], paraphrase=d["paraphrase"], variant=variant)
            except RuntimeError as exc:
                infeasible.append(f"{mech} seed{seed}: {exc}")
                continue
            if not cp.invariants.ok:
                infeasible.append(f"{mech} seed{seed}: invariants {cp.invariants.failures()}")
                continue
            for cond in conditions:
                for slot in slots:
                    mid = models_cfg["models"][slot]["id"]
                    for b in budgets:
                        for polarity in ("positive", "negative"):
                            rec = getattr(cp, polarity)
                            ph = digest_text(system_prompt(cond, b) + "||" + rec.task.question)
                            planned.append((make_run_id(rec.task.task_id, cond, mid, ph,
                                                        rec.world_hash),
                                            f"{mech} seed{seed} {cond} {slot} B{b} {polarity}"))

print(f"\n{len(planned)} episodes planned")
if infeasible:
    print(f"INFEASIBLE cells ({len(infeasible)}):")
    for f in infeasible[:10]:
        print("   ", f)

overlap = [(rid, lbl) for rid, lbl in planned if rid in existing]
fresh = [(rid, lbl) for rid, lbl in planned if rid not in existing]
print(f"  {len(fresh)} would run fresh (billed)")
print(f"  {len(overlap)} already exist on disk (would be duplicates)")
if overlap:
    src = Counter(existing[r] for r, _ in overlap)
    print(f"  duplicates of: {dict(src)}")
    for rid, lbl in overlap[:5]:
        print(f"    {rid}  {lbl}  <- {existing[rid]}")

bad = False
if args.expect == "new" and overlap:
    print("\nREFUSING: --expect new, but the plan reproduces run ids that already exist.\n"
          "The config change did not alter the worlds. Do NOT launch: this would re-bill an\n"
          "already-completed configuration (this is the M3 incident, exactly).")
    bad = True
if args.expect == "cached" and fresh:
    print(f"\nREFUSING: --expect cached, but {len(fresh)} episodes would be billed.")
    bad = True
if infeasible and args.expect != "report":
    print(f"\nREFUSING: {len(infeasible)} cells are infeasible under the current config.")
    bad = True
if not bad:
    print("\npreflight OK")
sys.exit(1 if bad else 0)
