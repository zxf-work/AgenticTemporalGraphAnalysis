"""Phase 0 acceptance test (T0.3).

Valid examples must pass; every documented invalid example must be rejected mechanically. Run:

    python scripts/selftest_schemas.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import SchemaError, stable_hash
from tasks.schema import (FORBIDDEN_SERVED_KEYS, Task, TaskRecord, make_task_id, validate_task,
                          validate_task_record)
from evaluation.claim_schema import Claim, extract_claim, validate_claim
from traces.schema import Step, Trace, make_run_id, validate_trace
from worlds.base import TAG_BACKGROUND, TAG_CONFOUNDER, TAG_MECHANISM, TemporalWorld
from worlds.invariants import check_pair

PASS, FAIL = "  ok  ", " FAIL "
_results: list[tuple[bool, str]] = []


def ok(name: str, cond: bool, detail: str = "") -> None:
    _results.append((cond, name))
    print(f"[{PASS if cond else FAIL}] {name}{(' — ' + detail) if detail and not cond else ''}")


def rejects(name: str, fn, *args, **kw) -> None:
    """Assert that fn raises SchemaError."""
    try:
        fn(*args, **kw)
    except SchemaError as exc:
        _results.append((True, name))
        print(f"[{PASS}] {name}  (rejected: {str(exc)[:70]})")
    except Exception as exc:  # wrong exception type is still a failure
        _results.append((False, name))
        print(f"[{FAIL}] {name} — raised {type(exc).__name__}, expected SchemaError")
    else:
        _results.append((False, name))
        print(f"[{FAIL}] {name} — accepted an invalid object")


# ======================================================================== tasks
print("\n--- task schema ---")
good_task = {
    "task_id": make_task_id("m1_none_seed000", "positive", 1),
    "question": "Investigate whether a time-respecting cascade links 4821 to 977.",
    "horizon": [0, 100000],
    "focus": {"source": 4821, "target": 977, "window": [41200, 44800], "max_wait": 600},
    "output_contract": "claim.v1",
}
validate_task(good_task)
ok("valid task accepted", True)
ok("task round-trips", Task.from_dict(good_task).to_dict() == good_task)
ok("task_id is opaque and deterministic",
   make_task_id("m1_none_seed000", "positive", 1) == good_task["task_id"]
   and make_task_id("m1_none_seed000", "negative", 1) != good_task["task_id"])

rejects("task with gold key", validate_task, {**good_task, "gold": True})
rejects("task with mechanism label", validate_task, {**good_task, "mechanism": "cascade"})
rejects("task with pair_id", validate_task, {**good_task, "pair_id": "m1_none_seed000"})
rejects("task missing decision contract", validate_task,
        {k: v for k, v in good_task.items() if k != "output_contract"})
rejects("task with window outside horizon", validate_task,
        {**good_task, "focus": {**good_task["focus"], "window": [41200, 999999]}})
rejects("task with inverted interval", validate_task,
        {**good_task, "focus": {**good_task["focus"], "window": [44800, 41200]}})
rejects("task with unknown focus key", validate_task,
        {**good_task, "focus": {**good_task["focus"], "planted_nodes": [1, 2]}})

good_record = {
    "task": good_task, "pair_id": "m1_none_seed000", "polarity": "positive",
    "mechanism": "cascade", "gold": True, "world_path": "data/worlds/m1_none_seed000_pos.csv",
    "world_hash": "abc123", "confounder": "none", "seed": 0, "paraphrase": 1,
}
validate_task_record(good_record)
ok("valid task record accepted", True)
rejects("record with polarity/gold contradiction", validate_task_record,
        {**good_record, "gold": False})
rejects("record with focus keys wrong for mechanism", validate_task_record,
        {**good_record, "mechanism": "closure_wave"})

# ======================================================================== claims
print("\n--- claim schema ---")
good_claim = {
    "task_id": good_task["task_id"], "decision": "present", "mechanism": "cascade",
    "entities": [4821, 977], "window": [41200, 44800], "confidence": 0.78,
    "evidence": [{"trace_id": "step_04", "role": "support"},
                 {"trace_id": "step_02", "role": "confounder_control"}],
    "narrative": "A time-respecting path exists with all waits under 600.",
}
validate_claim(good_claim)
ok("valid claim accepted", True)
c = Claim.from_dict(good_claim)
ok("claim round-trips", c.to_dict() == good_claim)
ok("claim helpers", c.is_positive and not c.is_abstention
   and c.cited("support") == ["step_04"] and c.cited("confounder_control") == ["step_02"])

rejects("claim with bad decision", validate_claim, {**good_claim, "decision": "maybe"})
rejects("claim with malformed trace id", validate_claim,
        {**good_claim, "evidence": [{"trace_id": "s4", "role": "support"}]})
rejects("claim with unknown evidence role", validate_claim,
        {**good_claim, "evidence": [{"trace_id": "step_04", "role": "vibes"}]})
rejects("claim with duplicate citation", validate_claim,
        {**good_claim, "evidence": [{"trace_id": "step_04", "role": "support"},
                                    {"trace_id": "step_04", "role": "support"}]})
rejects("claim with confidence out of range", validate_claim, {**good_claim, "confidence": 1.4})
rejects("positive claim with no entities", validate_claim, {**good_claim, "entities": []})
rejects("claim missing evidence key", validate_claim,
        {k: v for k, v in good_claim.items() if k != "evidence"})

print("\n--- claim extraction from model text ---")
ok("bare json", extract_claim('{"decision":"absent","x":1}')["decision"] == "absent")
ok("fenced json",
   extract_claim('Here you go:\n```json\n{"decision":"present"}\n```\nDone')["decision"]
   == "present")
ok("json after prose",
   extract_claim('I conclude the following. {"decision":"insufficient"}')["decision"]
   == "insufficient")
rejects("no json at all", extract_claim, "I could not determine an answer.")

# ======================================================================== traces
print("\n--- trace schema ---")
tr = Trace(run_id="r_x", task_id=good_task["task_id"], pair_id="m1_none_seed000",
           world_hash="w1", agent_condition="A3", model="test-model-v1", prompt_hash="p1",
           code_commit="deadbeef", tool_budget=20)
for i, tool in enumerate(["graph_metric_timeseries", "temporal_reachability"], start=1):
    tr.steps.append(Step(step_id=tr.next_step_id(), tool=tool, args={"a": i},
                         result_digest=stable_hash({"r": i}), result_summary=f"result {i}",
                         latency_ms=41, tokens_before=1900, tokens_after=2410))
validate_trace(tr.to_dict())
ok("valid trace accepted", True)
ok("step ids are contiguous", tr.step_ids() == {"step_01", "step_02"})
ok("trace round-trips", Trace.from_dict(tr.to_dict()).to_dict() == tr.to_dict())
ok("run id deterministic",
   make_run_id("t1", "A3", "m", "p", "w") == make_run_id("t1", "A3", "m", "p", "w")
   and make_run_id("t1", "A3", "m", "p", "w") != make_run_id("t1", "A1", "m", "p", "w"))

bad = tr.to_dict()
bad["steps"] = [bad["steps"][1]]           # step_02 without step_01
rejects("trace with non-contiguous step ids", validate_trace, bad)
rejects("trace with unknown agent condition", validate_trace,
        {**tr.to_dict(), "agent_condition": "A9"})

# ======================================================================== worlds
print("\n--- world representation ---")
rng = np.random.default_rng(0)
events = [(int(rng.integers(0, 50)), int(rng.integers(0, 50)), int(rng.integers(0, 1000)),
           TAG_BACKGROUND) for _ in range(400)]
events += [(1, 2, 100, TAG_MECHANISM), (2, 3, 200, TAG_MECHANISM)]
events += [(7, 8, 500, TAG_CONFOUNDER)]
w = TemporalWorld.from_events(50, events, horizon=(0, 1000))

ok("world hash is stable",
   w.world_hash() == TemporalWorld.from_events(50, list(events), (0, 1000)).world_hash())
shuffled = list(events)
rng.shuffle(shuffled)
ok("world hash ignores input row order",
   TemporalWorld.from_events(50, shuffled, (0, 1000)).world_hash() == w.world_hash())
ok("served view drops tags", w.served_events().shape == (len(events), 3))
ok("served view is canonically ordered",
   np.all(np.diff(w.served_events()[:, 2]) >= 0))
ok("tag counts internal only", w.tag_counts()["mechanism"] == 2)

# a timestamp-assignment permutation preserves every invariant surface
perm_events = list(events)
ts = [e[2] for e in perm_events]
rot = ts[1:] + ts[:1]
perm_events = [(u, v, t2, tag) for (u, v, _t, tag), t2 in zip(perm_events, rot)]
w2 = TemporalWorld.from_events(50, perm_events, horizon=(0, 1000))
ok("permutation preserves static edges", w.static_edge_multiset() == w2.static_edge_multiset())
ok("permutation preserves timestamp multiset", w.timestamp_multiset() == w2.timestamp_multiset())
ok("permutation preserves degree", bool(np.array_equal(w.degree_vector(), w2.degree_vector())))
ok("permutation changes the world hash", w.world_hash() != w2.world_hash())

print("\n--- invariant checker ---")
# stat = number of mechanism-tagged events whose times are increasing (a stand-in gold fn)
def toy_gold(world: TemporalWorld) -> tuple[bool, float]:
    m = world.tag == TAG_MECHANISM
    times = sorted(world.t[m].tolist())
    stat = 1.0 if times == world.t[m].tolist() else 0.0
    return stat > 0.5, stat

rec = check_pair(w, w2, pair_id="toy", mechanism="cascade", confounder="none", seed=0,
                 gold_fn=toy_gold, required_margin=0.4, threshold=0.5)
ok("checker confirms permutation_only", rec.permutation_only)
ok("checker confirms matched marginals",
   rec.same_static_edges and rec.same_timestamp_multiset and rec.same_degree_vector)

# a pair that is NOT a permutation must be caught
w3 = TemporalWorld.from_events(50, events + [(9, 9, 999, TAG_BACKGROUND)], horizon=(0, 1000))
rec_bad = check_pair(w, w3, pair_id="toy_bad", mechanism="cascade", confounder="none", seed=0,
                     gold_fn=toy_gold, required_margin=0.4, threshold=0.5)
ok("checker rejects non-permutation pair", not rec_bad.permutation_only and not rec_bad.ok,
   str(rec_bad.failures()))
ok("failure list is informative", "permutation_only" in rec_bad.failures())

# ======================================================================== summary
n_fail = sum(1 for good, _ in _results if not good)
print(f"\n{'=' * 62}\n{len(_results) - n_fail}/{len(_results)} checks passed")
if n_fail:
    print("FAILED: " + ", ".join(name for good, name in _results if not good))
sys.exit(1 if n_fail else 0)
