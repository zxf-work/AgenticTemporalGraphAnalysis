"""Milestone C6 — measure backend cost by replaying recorded tool calls. No LLM, no API cost.

    python scripts/replay_backend.py runs/a2_opus runs/resolution_opus ...
    python scripts/replay_backend.py --all

Every trace records the exact tool name and arguments of every call it made, and the worlds
recompile deterministically, so the data-system half of the cost can be measured *after the fact*
by re-executing those calls against the same world. That matters because backend accounting was
added after most of the matrix had already run: without replay, C6 would either report zeros for
completed work or demand an expensive re-run of finished experiments.

What this measures, per run:

    rows_scanned   events the operators examined  (A2: the whole log per call, since the entire
                   event list is handed to the interpreter regardless of what the program does)
    backend_ms     wall-clock inside the operator implementations, replayed serially

Replay is strictly serial and on an idle-ish host, which is what makes these numbers comparable:
the sweeps themselves ran at concurrency 8-12, so their recorded step latencies include queueing
and contention and must never be used as backend-cost evidence. C6 says this explicitly.

Writes `backend_cost.jsonl` into each run directory. Nothing else is modified.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.tools_spec import surface_for
from tasks.compiler import load_world
from tasks.lookup import record_for
from traces.schema import Trace

args = [a for a in sys.argv[1:] if not a.startswith("--")]
if "--all" in sys.argv or not args:
    dirs = sorted(d for d in (ROOT / "runs").glob("*") if (d / "results.jsonl").exists())
else:
    dirs = [pathlib.Path(a) for a in args]

mcfg = yaml.safe_load((ROOT / "configs/mechanisms.yaml").read_text())
xcfg = yaml.safe_load((ROOT / "configs/experiment_matrix.yaml").read_text())
cache: dict[tuple, object] = {}

for d in dirs:
    rows = {}
    for line in (d / "results.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r.get("run_id")] = r

    out, skipped = [], 0
    for run_id, row in rows.items():
        tp = d / "traces" / f"{run_id}.json"
        if not tp.exists() or row.get("model") == "script-a1-bisect":
            skipped += 1
            continue
        try:
            trace = Trace.from_dict(json.loads(tp.read_text()))
            rec = record_for(row, mcfg, xcfg, cache)
            if rec.world_hash != trace.world_hash:
                skipped += 1
                continue
            world = load_world(rec.world_path)
            by_name = {t.name: t for t in surface_for(row["condition"])}
        except Exception as exc:                                   # noqa: BLE001
            print(f"  ! {d.name}/{run_id}: {type(exc).__name__}: {str(exc)[:80]}")
            skipped += 1
            continue

        rows_scanned, backend_ms, replayed, failed = 0, 0.0, 0, 0
        for step in trace.steps:
            spec = by_name.get(step.tool)
            if spec is None:
                continue
            before = world.scanned
            t0 = time.perf_counter()
            try:
                result = spec.fn(world, **step.args)
            except Exception:                                      # noqa: BLE001
                # A call that errored originally errors again; it still cost backend work, and
                # dropping it would understate cost for exactly the runs that struggled.
                result = None
                failed += 1
            backend_ms += (time.perf_counter() - t0) * 1000.0
            n = world.scanned - before
            if step.tool == "run_python" and isinstance(result, dict):
                n = max(n, int(result.get("n_events", 0)))
            rows_scanned += n
            replayed += 1

        out.append({
            "run_id": run_id, "condition": row["condition"], "model": row["model"],
            "mechanism": row["mechanism"], "budget": row.get("budget"),
            "sweep": row.get("sweep", ""), "polarity": row["polarity"],
            "tool_calls": len(trace.steps), "llm_calls": trace.llm_calls,
            "input_tokens": trace.total_input_tokens, "output_tokens": trace.total_output_tokens,
            "cost_usd": row.get("cost_usd", 0.0),
            "rows_scanned": rows_scanned, "backend_ms": round(backend_ms, 2),
            "replayed_calls": replayed, "replay_errors": failed,
        })

    (d / "backend_cost.jsonl").write_text("".join(json.dumps(o) + "\n" for o in out))
    tot_rows = sum(o["rows_scanned"] for o in out)
    print(f"{d.name}: replayed {len(out)} runs ({skipped} skipped), "
          f"{tot_rows:,} events scanned, {sum(o['backend_ms'] for o in out) / 1000:.1f}s backend")
