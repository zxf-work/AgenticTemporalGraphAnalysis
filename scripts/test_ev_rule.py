"""The evidence-validity rule must be the one the paper states, and the old one must stay reachable.

Two things are checked.

**The rule itself**, on hand-built traces: the stated rule is existential over cited support, so
an off-bound call cited beside the task-bound one does not fail a run; an empty support set, a
nonexistent-step citation and a wrong decision each do. The submitted rule, kept for
reproducibility, is stricter on A3 for M1-M3 and is asserted to differ exactly there.

**The corpus**, end to end: the headline A3 cells of the paper's Table 3 under both rules, the
agreement of the two code paths that compute the stated rule (`RunRow.ev_valid` from a fresh
`score()`, and `figdata.ev_valid` from a stored row), and the count of rows on which the rules
disagree.

Run:  python scripts/test_ev_rule.py
"""

from __future__ import annotations

import pathlib
import sys
from collections import Counter

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import stable_hash
from evaluation import figdata
from evaluation.ev_rule import STATED, SUBMITTED, get_ev_rule, set_ev_rule
from evaluation.exec_support import cited_ev
from evaluation.score_run import score
from tasks.compiler import compile_pair
from traces.schema import Step, Trace

cfg = yaml.safe_load((ROOT / "configs/mechanisms.yaml").read_text())
results: list[tuple[bool, str]] = []


def ok(name: str, cond: bool, detail: str = "") -> None:
    results.append((cond, name))
    print(f"[{'  ok  ' if cond else ' FAIL '}] {name}{('  ' + detail) if detail else ''}")


cp = compile_pair(cfg, "cascade", 0, scale="dev", write=False)
rec = cp.positive
f = rec.task.focus
BOUND = {"source": f["source"], "target": f["target"], "window": f["window"],
         "max_wait": f["max_wait"]}
UNBOUND = {"source": f["source"], "target": f["target"], "window": f["window"]}


def trace_with(calls, evidence, *, decision="present", condition="A3", ok_steps=True) -> Trace:
    t = Trace(run_id="r", task_id=rec.task.task_id, pair_id=rec.pair_id,
              world_hash=rec.world_hash, agent_condition=condition, model="test",
              prompt_hash="p", code_commit="", tool_budget=20)
    for tool, args in calls:
        t.steps.append(Step(step_id=t.next_step_id(), tool=tool, args=args,
                            result_digest=stable_hash(args), result_summary="{}",
                            latency_ms=1, tokens_before=0, tokens_after=0, ok=ok_steps))
    t.final_claim = {"task_id": rec.task.task_id, "decision": decision, "mechanism": "cascade",
                     "entities": [f["source"]], "window": f["window"], "confidence": 0.8,
                     "evidence": [{"trace_id": i, "role": role} for i, role in evidence]}
    return t


def both(trace: Trace, condition: str = "A3") -> tuple[bool, bool]:
    """(stated, submitted) verdict of a freshly scored row, plus a check of `cited_ev`."""
    row = score(trace, rec, condition)
    set_ev_rule(STATED)
    stated = row.ev_valid
    assert stated == cited_ev(trace, rec.task, "cascade", condition, rec.gold)
    set_ev_rule(SUBMITTED)
    submitted = row.ev_valid
    set_ev_rule(STATED)
    return stated, submitted


print("--- the stated rule is existential over cited support ---")
st, sub = both(trace_with([("temporal_reachability", BOUND)], [("step_01", "support")]))
ok("task-bound call cited alone: valid under both rules", st and sub)

st, sub = both(trace_with([("temporal_reachability", UNBOUND), ("temporal_reachability", BOUND)],
                          [("step_01", "support"), ("step_02", "support")]))
ok("off-bound call cited beside the task-bound one: valid under the stated rule", st)
ok("...and invalid under the submitted rule (the difference being corrected)", not sub)

st, sub = both(trace_with([("temporal_reachability", UNBOUND)], [("step_01", "support")]))
ok("only an off-bound call cited: invalid under both rules", not st and not sub)

st, sub = both(trace_with([("temporal_reachability", UNBOUND), ("temporal_reachability", BOUND)],
                          [("step_01", "support"), ("step_02", "confounder_control")]))
ok("task-bound call cited only as a confounder control does not count as support", not st)

print("\n--- what fails a run under the stated rule ---")
st, _ = both(trace_with([("temporal_reachability", BOUND)],
                        [("step_01", "support"), ("step_09", "support")]))
ok("a nonexistent-step citation beside valid support is invalid", not st)

neg = cp.negative


def neg_trace(evidence) -> Trace:
    t = trace_with([("temporal_reachability", BOUND)], evidence, decision="absent")
    t.task_id, t.world_hash = neg.task.task_id, neg.world_hash
    t.final_claim["task_id"] = neg.task.task_id
    return t


row = score(neg_trace([]), neg, "A3")
set_ev_rule(SUBMITTED)
sub = row.ev_valid
set_ev_rule(STATED)
ok("an absent claim with an empty support set is invalid under the stated rule",
   not row.ev_valid)
ok("...and was passed by the submitted rule", sub)
row = score(neg_trace([("step_01", "support")]), neg, "A3")
ok("an absent claim citing the task-bound call is valid", row.ev_valid)

st, _ = both(trace_with([("temporal_reachability", BOUND)], [("step_01", "support")],
                        decision="absent"))
ok("a wrong decision is invalid whatever it cites", not st)
st, _ = both(trace_with([("temporal_reachability", BOUND)], [("step_01", "support")],
                        decision="insufficient"))
ok("an abstention is invalid", not st)

print("\n--- A1: no operation takes a task-constraint argument ---")
st, sub = both(trace_with([("window_event_count", {"windows": [f["window"]]})],
                          [("step_01", "support")], condition="A1"), "A1")
ok("any successfully executed cited support step passes", st and sub)
st, _ = both(trace_with([("window_event_count", {"windows": [f["window"]]})],
                        [("step_01", "support")], condition="A1", ok_steps=False), "A1")
ok("a cited step that errored does not", not st)
st, _ = both(trace_with([("window_event_count", {"windows": [f["window"]]})],
                        [("step_01", "confounder_control")], condition="A1"), "A1")
ok("an A1 run with no support citation is invalid", not st)

# ======================================================================================
print("\n--- corpus: Table 3 headline cells under both rules ---")
ok("default rule is the stated one", get_ev_rule() == STATED)
rows = figdata.load()
HEAD = {"cascade": ("resolution=32", "resolution=128", "resolution=512"),
        "closure_wave": ("cross_mechanism=closure_wave",),
        "regime_switch": ("cross_mechanism=regime_switch",),
        "compound": ("compound=compound",)}
#: (mechanism, model) -> (EVPairAcc stated, EVPairAcc submitted); PairAcc is 1.00 throughout
EXPECT = {("cascade", "claude-opus-5"): (1.00, 0.67), ("cascade", "gpt-5-mini"): (1.00, 0.53),
          ("closure_wave", "claude-opus-5"): (0.80, 0.20),
          ("closure_wave", "gpt-5-mini"): (1.00, 0.80),
          ("regime_switch", "claude-opus-5"): (1.00, 0.60),
          ("regime_switch", "gpt-5-mini"): (1.00, 1.00),
          ("compound", "claude-opus-5"): (0.20, 0.20), ("compound", "gpt-5-mini"): (0.00, 0.00)}
for (mech, model), (want_st, want_sub) in EXPECT.items():
    sel = [r for r in rows if r["mechanism"] == mech and r["condition"] == "A3"
           and r["model"] == model and r.get("sweep") in HEAD[mech]
           and r.get("confounder", "none") == "none" and r.get("scale", "dev") == "dev"]
    set_ev_rule(STATED)
    pa, ev_st, n, _ = figdata.paired(sel)
    set_ev_rule(SUBMITTED)
    _, ev_sub, _, _ = figdata.paired(sel)
    set_ev_rule(STATED)
    ok(f"{mech:<14}{figdata.MODELS[model]:<11} n={n:<3} PairAcc 1.00, EV {want_st:.2f} stated / "
       f"{want_sub:.2f} submitted",
       pa == 1.0 and round(ev_st, 2) == want_st and round(ev_sub, 2) == want_sub,
       f"got {pa:.2f} / {ev_st:.2f} / {ev_sub:.2f}")

print("\n--- corpus: every row, both code paths ---")
agree, differ = 0, Counter()
for r in rows:
    trace, trec, tc = figdata.trace_and_record(r)
    fresh = score(trace, trec, r["condition"], compound_tc=tc)
    st = figdata.ev_valid(r, STATED)
    agree += fresh.ev_valid == st and fresh.ev_valid_submitted == figdata.ev_valid(r, SUBMITTED)
    if st != figdata.ev_valid(r, SUBMITTED):
        differ[r["condition"]] += 1
ok(f"all {len(rows)} rows have a stored trace on a world that recompiles", len(rows) == 1402)
ok("RunRow.ev_valid (fresh score) equals figdata.ev_valid (stored row) under both rules",
   agree == len(rows), f"{agree}/{len(rows)}")
ok("the rules disagree on 58 A3 rows and 2 A1 rows, and on no A0 or A2 row",
   dict(differ) == {"A3": 58, "A1": 2}, str(dict(differ)))
m4 = [r for r in rows if r["mechanism"] == "compound"]
ok("...and on no M4 row", all(figdata.ev_valid(r, STATED) == figdata.ev_valid(r, SUBMITTED)
                              for r in m4), f"{len(m4)} M4 rows")

n_fail = sum(1 for good_, _ in results if not good_)
print(f"\n{'=' * 66}\n{len(results) - n_fail}/{len(results)} checks passed")
if n_fail:
    print("FAILED: " + ", ".join(n for g, n in results if not g))
sys.exit(1 if n_fail else 0)
