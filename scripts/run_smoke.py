"""End-to-end smoke test on the mock backend (no API key required).

Proves the pipeline: generate pair -> compile task -> run agent loop -> dispatch tools -> log trace
-> extract claim -> check evidence -> aggregate paired metrics.

Also validates the R4 chance floors empirically, which is what lets the paper state them as facts
rather than as arithmetic claims:

    always_present  -> 0.00 pair accuracy
    always_absent   -> 0.00
    coin_flip       -> ~0.25
    static_oracle   -> 0.00   (forced: the static surface is identical across a pair)

Run:  python scripts/run_smoke.py [n_pairs]
"""

from __future__ import annotations

import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.harness import MockBackend, run_episode
from evaluation.checker import check
from evaluation.metrics import RunRow, expected_floor, summarize
from tasks.compiler import compile_pair, load_world

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
cfg = yaml.safe_load((ROOT / "configs/mechanisms.yaml").read_text())
results: list[tuple[bool, str]] = []


def ok(name: str, cond: bool, detail: str = "") -> None:
    results.append((cond, name))
    print(f"[{'  ok  ' if cond else ' FAIL '}] {name}{('  ' + detail) if detail else ''}")


print(f"compiling {N} cascade pairs...")
pairs = [compile_pair(cfg, "cascade", s, scale="dev") for s in range(N)]
worlds = {}
for cp in pairs:
    for rec in (cp.positive, cp.negative):
        worlds[rec.task.task_id] = (rec, load_world(rec.world_path))


def reach_script(rec) -> list[dict]:
    """One temporal_reachability call parameterised by *this* task's focus."""
    f = rec.task.focus
    return [{"name": "temporal_reachability",
             "args": {"source": f["source"], "target": f["target"], "window": f["window"],
                      "max_wait": f["max_wait"]}}]


def summary_script(rec) -> list[dict]:
    """One order-blind call — the kind of evidence a static agent would cite as support."""
    return [{"name": "graph_summary", "args": {"window": rec.task.focus["window"]}}]


def run_policy(policy: str, condition: str, script_fn=None) -> list[RunRow]:
    rows = []
    for tid, (rec, world) in worlds.items():
        backend = MockBackend(policy=policy, seed=7,
                              script=(script_fn(rec) if script_fn else []))
        tr = run_episode(task=rec.task, world=world, condition=condition, backend=backend,
                         pair_id=rec.pair_id, world_hash=rec.world_hash, tool_budget=20)
        res = check(tr, rec.gold, condition=condition, horizon=rec.task.horizon)
        rows.append(RunRow(
            pair_id=rec.pair_id, polarity=rec.polarity, gold=rec.gold, condition=condition,
            model=backend.name, mechanism=rec.mechanism, confounder=rec.confounder,
            decision=(tr.final_claim or {}).get("decision", "none"), check=res,
            tool_calls=len(tr.steps), input_tokens=tr.total_input_tokens,
            output_tokens=tr.total_output_tokens, latency_ms=tr.total_latency_ms,
            status=tr.status))
    return rows


# ---------------------------------------------------------------- 1. the loop works
print("\n--- pipeline: scripted tool use then answer (A3 surface) ---")
rows = run_policy("tool_then_answer", "A3", script_fn=reach_script)
s = summarize(rows)
ok("every run produced a claim", all(r.decision != "none" for r in rows))
ok("tools were actually dispatched", all(r.tool_calls == 1 for r in rows),
   f"tool_calls={sorted({r.tool_calls for r in rows})}")
ok("an agent that reads temporal_reachability gets every pair right", s.pair_acc == 1.0,
   f"pair_acc={s.pair_acc:.2f}")
ok("citations resolve to real steps", all(r.check.citations_exist for r in rows))

# ---------------------------------------------------------------- 2. chance floors (R4)
print("\n--- R4 chance floors ---")
for policy in ("always_present", "always_absent", "coin_flip", "static_oracle"):
    rows = run_policy(policy, "A3")
    s = summarize(rows)
    exp = expected_floor(policy)
    if policy == "coin_flip":
        good = abs(s.pair_acc - exp) <= 0.25            # small N, wide tolerance
    else:
        good = s.pair_acc == exp
    ok(f"{policy:16s} pair_acc={s.pair_acc:.2f} (expected {exp:.2f})", good,
       f"single_world={s.single_world_acc:.2f} consistency={s.consistency}")

# ---------------------------------------------------------------- 3. the A1 impossibility
print("\n--- the A1 surface cannot separate a pair, by construction ---")
rows = run_policy("static_oracle", "A1")
s = summarize(rows)
ok("static-only agent scores 0 pair accuracy", s.pair_acc == 0.0, f"pair_acc={s.pair_acc:.2f}")
ok("...while its single-world accuracy looks respectable",
   abs(s.single_world_acc - 0.5) < 1e-9, f"single_world_acc={s.single_world_acc:.2f}")

# ---------------------------------------------------------------- 4. failure paths
print("\n--- degenerate model behaviour is handled ---")
rows = run_policy("malformed", "A3")
ok("malformed output is recorded, not crashed",
   all(r.status == "malformed_output" for r in rows),
   f"statuses={sorted({r.status for r in rows})}")
ok("malformed runs score as incorrect", all(not r.check.decision_correct for r in rows))

# ---------------------------------------------------------------- 5. A4 evidence policy
print("\n--- A4 requires a confounder control for a positive claim ---")
# an agent that cites supporting evidence but never tests an alternative explanation: A3 accepts
# it, A4 must not
rows_a3 = run_policy("always_present", "A3", script_fn=summary_script)
rows_a4 = run_policy("always_present", "A4", script_fn=summary_script)
upr_a3 = summarize(rows_a3).unsupported_positive_rate
upr_a4 = summarize(rows_a4).unsupported_positive_rate
ok("A3 accepts a positive backed by support alone", upr_a3 == 0.0, f"UPR A3={upr_a3:.2f}")
ok("A4 flags the same claim as uncontrolled", upr_a4 == 1.0, f"UPR A4={upr_a4:.2f}")

# and with no citations at all, both conditions must flag it
rows_bare = run_policy("always_present", "A3")
ok("a positive with no citations is unsupported under A3 too",
   summarize(rows_bare).unsupported_positive_rate == 1.0)

n_fail = sum(1 for good, _ in results if not good)
print(f"\n{'=' * 62}\n{len(results) - n_fail}/{len(results)} checks passed")
if n_fail:
    print("FAILED: " + ", ".join(n for g, n in results if not g))
sys.exit(1 if n_fail else 0)
