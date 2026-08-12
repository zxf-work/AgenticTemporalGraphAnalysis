"""Operators that answer the same question must agree, and the parameterisation metric must fire.

Two things are checked.

**Operator agreement.** ``temporal_reachability`` and ``temporal_paths`` decide the same
feasibility question. A development run exposed a disagreement — the source-departure exemption
(D7) had been applied to reachability but not to paths, so the same task got opposite answers
depending on which operator the agent happened to call. That is a defect that would have been
scored as a *model* error, so it is now asserted.

**Parameterisation detection.** Deliberately mis-aimed calls must be caught by the reported
metric, and correctly-aimed ones must not raise false positives.

Run:  python scripts/test_operator_consistency.py [n_pairs]
"""

from __future__ import annotations

import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import stable_hash
from evaluation.error_taxonomy import detects_range_reconstruction, parameterisation_report
from tasks.compiler import compile_pair, load_world
from tools.temporal_surface import temporal_paths, temporal_reachability
from traces.schema import Step, Trace

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
cfg = yaml.safe_load((ROOT / "configs/mechanisms.yaml").read_text())
results: list[tuple[bool, str]] = []


def ok(name: str, cond: bool, detail: str = "") -> None:
    results.append((cond, name))
    print(f"[{'  ok  ' if cond else ' FAIL '}] {name}{('  ' + detail) if detail else ''}")


print("--- reachability and paths must agree on feasibility ---")
disagreements = []
for seed in range(N):
    cp = compile_pair(cfg, "cascade", seed, scale="dev")
    f = cp.positive.task.focus
    for rec in (cp.positive, cp.negative):
        w = load_world(rec.world_path)
        r = temporal_reachability(w, f["source"], f["target"], window=tuple(f["window"]),
                                  max_wait=f["max_wait"])
        p = temporal_paths(w, f["source"], f["target"], window=tuple(f["window"]),
                           max_wait=f["max_wait"], max_paths=1)
        if r.reached != bool(p):
            disagreements.append((rec.pair_id, rec.polarity, r.reached, bool(p)))
ok(f"agreement over {N} pairs x 2 polarities", not disagreements,
   f"disagreements: {disagreements[:4]}")

print("\n--- every path returned must respect the waiting bound ---")
violations = []
for seed in range(min(N, 4)):
    cp = compile_pair(cfg, "cascade", seed, scale="dev")
    f = cp.positive.task.focus
    w = load_world(cp.positive.world_path)
    for path in temporal_paths(w, f["source"], f["target"], window=tuple(f["window"]),
                               max_wait=f["max_wait"], max_paths=3):
        times = [t for _u, _v, t in path]
        if any(b <= a for a, b in zip(times, times[1:])):
            violations.append(("not increasing", path))
        # the bound applies between consecutive hops, never to the source's departure
        if any((b - a) > f["max_wait"] for a, b in zip(times, times[1:])):
            violations.append(("gap exceeds max_wait", path))
ok("returned paths are time-respecting and within the bound", not violations,
   str(violations[:2]))

# ======================================================================================
print("\n--- parameterisation metric ---")
cp = compile_pair(cfg, "cascade", 0, scale="dev")
rec = cp.positive
f = rec.task.focus


def trace_with(*calls) -> Trace:
    t = Trace(run_id="r", task_id=rec.task.task_id, pair_id=rec.pair_id, world_hash="w",
              agent_condition="A3", model="test", prompt_hash="p", code_commit="", tool_budget=20)
    for tool, args in calls:
        t.steps.append(Step(step_id=t.next_step_id(), tool=tool, args=args,
                            result_digest=stable_hash(args), result_summary="{}",
                            latency_ms=1, tokens_before=0, tokens_after=0))
    t.final_claim = {"task_id": rec.task.task_id, "decision": "present", "mechanism": "cascade",
                     "entities": [f["source"]], "window": f["window"], "confidence": 0.8,
                     "evidence": [{"trace_id": "step_01", "role": "support"}]}
    return t


good = trace_with(("temporal_reachability", {"source": f["source"], "target": f["target"],
                                             "window": f["window"], "max_wait": f["max_wait"]}))
rep = parameterisation_report(good, rec.task, "cascade")
ok("correctly aimed call raises no error", not rep.any_error, str(rep.details))
ok("...and is not flagged as bad support", not rep.cited_bad_support)

miss = trace_with(("temporal_reachability", {"source": f["source"], "target": f["target"],
                                             "window": f["window"]}))
rep = parameterisation_report(miss, rec.task, "cascade")
ok("omitted max_wait is caught", rep.missing_constraint == 1, str(rep.details))
ok("...and flagged because the conclusion cites it", rep.cited_bad_support)

wrongc = trace_with(("temporal_reachability", {"source": f["source"], "target": f["target"],
                                               "window": f["window"], "max_wait": 99999}))
ok("wrong max_wait is caught",
   parameterisation_report(wrongc, rec.task, "cascade").wrong_constraint == 1)

narrow = trace_with(("temporal_reachability",
                     {"source": f["source"], "target": f["target"], "max_wait": f["max_wait"],
                      "window": [f["window"][0] + 100, f["window"][1] - 100]}))
ok("too-narrow window is caught",
   parameterisation_report(narrow, rec.task, "cascade").scope_mismatch == 1)

wrongn = trace_with(("temporal_reachability", {"source": 999999, "target": 999998,
                                               "window": f["window"],
                                               "max_wait": f["max_wait"]}))
ok("wrong entities are caught",
   parameterisation_report(wrongn, rec.task, "cascade").wrong_entities == 1)

print("\n--- range-reconstruction detection ---")
recon = trace_with(*[("window_event_count", {"nodes": [f["source"]],
                                             "windows": [[0, 10], [11, 20], [21, 30]]})
                     for _ in range(4)])
ok("batched node-filtered range search is detected", detects_range_reconstruction(recon))
plain = trace_with(("window_event_count", {"windows": [[0, 10]]}))
ok("a single unfiltered range call is not", not detects_range_reconstruction(plain))

n_fail = sum(1 for good_, _ in results if not good_)
print(f"\n{'=' * 66}\n{len(results) - n_fail}/{len(results)} checks passed")
if n_fail:
    print("FAILED: " + ", ".join(n for g, n in results if not g))
sys.exit(1 if n_fail else 0)
