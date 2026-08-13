"""Failure taxonomy and A3 parameterisation errors.

Two things live here.

**Parameterisation errors (reported metric).** The obvious objection to the semantic-operator
interface is that its operators "encode the answer": if `temporal_reachability` decides the task,
A3's advantage is trivial. The honest reply is not a rebuttal but a measurement — the agent still
has to *aim* the operator, and it can miss. We report how often A3 invokes a temporal operator
with a scope or constraint that does not match the task, and, more importantly, how often a
conclusion **cites a mis-parameterised result as its support**. The second is the number that
matters: it is the rate at which A3's answers rest on evidence that does not actually address the
question asked.

**Failure taxonomy (plan §18).** Rule-based classification of what went wrong, so failures can be
reported as a distribution rather than anecdotally.
"""

from __future__ import annotations

import pathlib
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tasks.schema import Task
from traces.schema import Trace

#: operators whose parameters are checkable against the task focus
TEMPORAL_OPS = {"temporal_reachability", "temporal_paths", "temporal_closure",
                "neighborhood_evolution"}


@dataclass
class ParamReport:
    n_temporal_calls: int = 0
    scope_mismatch: int = 0          # window does not cover the interval the task asks about
    missing_constraint: int = 0      # the task states a bound (max_wait / delta) and the call omits it
    wrong_constraint: int = 0        # a bound is given but differs from the task's
    wrong_entities: int = 0          # source/target/group not the ones under investigation
    cited_bad_support: bool = False  # a *cited* support step asked a different question
    details: list[str] = field(default_factory=list)

    # ---- what counts as an error ------------------------------------------------------
    # Two things looked alike in the first version and are not:
    #
    #   * changing the CONSTRAINT the task states (omitting the waiting bound, or using a
    #     different one) makes the operator answer a *different question* — unconstrained
    #     reachability is not Δ-constrained reachability, so a conclusion resting on it does not
    #     address the task. That is a genuine parameterisation error.
    #   * narrowing the WINDOW, or aiming at an intermediate node, is ordinary decomposition —
    #     an agent locating a cascade legitimately probes sub-intervals and waypoints. Counting
    #     that as error inflated the rate and would have reported competent analysis as a defect.
    #
    # Only the first drives `error_rate` and `cited_bad_support`; the second is reported
    # separately as `decomposition_rate` so the behaviour is still visible.

    @property
    def any_error(self) -> bool:
        return bool(self.missing_constraint or self.wrong_constraint)

    @property
    def error_rate(self) -> float:
        bad = self.missing_constraint + self.wrong_constraint
        return bad / self.n_temporal_calls if self.n_temporal_calls else 0.0

    @property
    def decomposition_rate(self) -> float:
        d = self.scope_mismatch + self.wrong_entities
        return d / self.n_temporal_calls if self.n_temporal_calls else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"n_temporal_calls": self.n_temporal_calls, "scope_mismatch": self.scope_mismatch,
                "missing_constraint": self.missing_constraint,
                "wrong_constraint": self.wrong_constraint, "wrong_entities": self.wrong_entities,
                "cited_bad_support": self.cited_bad_support,
                "error_rate": round(self.error_rate, 4),
                "decomposition_rate": round(self.decomposition_rate, 4),
                "details": self.details[:10]}


#: the constraint each mechanism's question states, and which an answer must actually be bound by
CONSTRAINT_KEY = {"cascade": "max_wait", "closure_wave": "delta", "regime_switch": "bandwidth",
                  "compound": "max_wait"}


def _compound_stage2_anchored(step: Any, focus: dict[str, Any], t_c: int,
                              claim_positive: bool) -> bool:
    """Anchored, claim-sensitive stage-2 test, used when the true ``t_c`` is available.

    The oracle's closure-completion time is computable at scoring time from the world itself
    (worlds.mechanisms.compound.closure_completion_time), so the checker can verify that a cited
    stage-2 query is tied to the admissible interval ``[t_c, t_c + admissible]`` rather than
    merely narrow. The window must lie inside the admissible interval padded by 25% of the lag on
    each side, matching the slack of the width-only proxy below. For a positive claim, any
    witness inside such a window lies within the (padded) admissible interval. For an absence
    claim the window must additionally cover the entire admissible interval, so a negative
    reachability result establishes absence over all of it.
    """
    adm = focus.get("admissible")
    if adm is None:
        return True                       # nothing to check against; never accuse by default
    w = step.args.get("window")
    if not (isinstance(w, (list, tuple)) and len(w) == 2):
        return False                      # no window at all == the whole observation period
    try:
        lo, hi = int(w[0]), int(w[1])
    except (TypeError, ValueError):
        return False
    pad = int(adm) * 0.25
    inside = (lo >= t_c - pad) and (hi <= t_c + int(adm) + pad)
    if claim_positive:
        return inside
    return inside and lo <= t_c and hi >= t_c + int(adm)


def _compound_stage2_bound(step: Any, focus: dict[str, Any]) -> bool:
    """Did this reachability call bind stage 2 to the *result* of stage 1?

    The compound question is decided by whether the cascade begins inside
    ``[t_c, t_c + admissible]``. An agent that runs reachability over the whole observation window
    answers a different question — "is there a cascade anywhere?" — which is TRUE in both worlds by
    construction, so a correct final answer on top of it is unsupported by its own evidence.

    ``t_c`` is never in the served focus (computing it is the compositional step under test), so
    this width-only **proxy** is what a caller without scoring-time world access can check: the
    call must query a strict sub-interval of the observation window, no wider than the admissible
    lag (with 50% slack for an agent that pads sensibly). Reported metrics use the anchored rule
    above, which recomputes the true t_c from the world at scoring time.
    """
    w = step.args.get("window")
    fwin = focus.get("window") or []
    adm = focus.get("admissible")
    if adm is None or len(fwin) != 2:
        return True                       # nothing to check against; never accuse by default
    if not (isinstance(w, (list, tuple)) and len(w) == 2):
        return False                      # no window at all == the whole observation period
    try:
        lo, hi = int(w[0]), int(w[1])
    except (TypeError, ValueError):
        return False
    narrowed = lo > int(fwin[0]) or hi < int(fwin[1])
    right_shape = (hi - lo) <= int(adm) * 1.5
    return narrowed and right_shape


def _a2_constraint_report(trace: Trace, task: Task, mechanism: str) -> ParamReport:
    """The A2 analogue of the A3 parameterisation check.

    Without this, EVPairAcc is not comparable across interfaces and would flatter A2: the
    `cited_bad_support` test fires only on semantic operators, so an A2 run scores a perfect
    evidence-validity gap **by construction** rather than by virtue. A2's characteristic failure
    is different in form but identical in consequence -- a program that never binds the bound the
    task states computes unconstrained reachability and answers a different question, exactly as
    an unparameterised operator does.

    The check is deterministic and deliberately generous, because a false accusation is worse than
    a miss here: the program passes if it mentions the stated constraint value anywhere, or reads
    it from the `focus` dict the sandbox provides. It therefore catches the clear case -- code
    that never refers to the bound at all -- and stays silent otherwise. It cannot verify that the
    constraint was applied *correctly*; that limitation is reported rather than papered over.
    """
    rep = ParamReport()
    key = CONSTRAINT_KEY.get(mechanism)
    want = task.focus.get(key) if key else None
    if want is None:
        return rep
    cited_support = {e["trace_id"] for e in (trace.final_claim or {}).get("evidence", [])
                     if e.get("role") == "support"}
    cited_bound, cited_any = False, False
    for step in trace.steps:
        if step.tool != "run_python" or not step.ok:
            continue
        code = str(step.args.get("code", ""))
        rep.n_temporal_calls += 1
        bound = (str(int(want)) in code) or (key in code)
        if not bound:
            rep.missing_constraint += 1
            rep.details.append(f"{step.step_id} run_python: program never references the stated "
                               f"{key} ({want})")
        if step.step_id in cited_support:
            cited_any = True
            cited_bound |= bound

    # A conclusion is unsupported only when NONE of its cited programs bound the constraint.
    # Flagging a run because *some* cited program was unbound punished ordinary reconnaissance:
    # agents routinely cite a data-dump ("how many events, which edges touch the endpoints")
    # alongside the constrained computation that actually decides. The A3 check ignores static
    # calls for exactly this reason and only inspects temporal operators; this is the A2
    # equivalent. Measured on Opus, the stricter rule reported a 0.76 evidence gap that dissolved
    # on inspection -- the decisive program was right there in the citation list.
    rep.cited_bad_support = cited_any and not cited_bound
    return rep


def parameterisation_report(trace: Trace, task: Task, mechanism: str,
                            compound_tc: int | None = None) -> ParamReport:
    """Check every temporal-operator call in a trace against what the task actually asked.

    ``compound_tc`` is the oracle closure-completion time for this world. When provided, M4
    stage-2 citations are judged by the anchored claim-sensitive rule; otherwise the width-only
    proxy applies.
    """
    if trace.agent_condition == "A2":
        return _a2_constraint_report(trace, task, mechanism)
    rep = ParamReport()
    focus = task.focus
    claim_positive = (trace.final_claim or {}).get("decision") == "present"

    def stage2_ok(step: Any) -> bool:
        if compound_tc is not None:
            return _compound_stage2_anchored(step, focus, compound_tc, claim_positive)
        return _compound_stage2_bound(step, focus)
    fwin = tuple(focus.get("window") or task.horizon)
    cited_support = {e["trace_id"] for e in (trace.final_claim or {}).get("evidence", [])
                     if e.get("role") == "support"}

    # Compound needs the same rule A2 needed: reconnaissance-then-refine is the natural workflow,
    # so a run must be judged on whether ANY cited call bound stage 2 -- not on whether some cited
    # call was broad. Judging per-call reported a 1.00 evidence gap for both models on M4, when
    # inspection showed Opus citing a correctly bound [t_c, t_c+admissible] query alongside a wide
    # exploratory one. Tracked here and resolved after the loop.
    stage2_cited_any = False
    stage2_cited_bound = False

    for step in trace.steps:
        if step.tool not in TEMPORAL_OPS or not step.ok:
            continue
        rep.n_temporal_calls += 1
        bad: list[str] = []
        if mechanism == "compound" and step.tool in ("temporal_reachability", "temporal_paths"):
            if step.step_id in cited_support:
                stage2_cited_any = True
                stage2_cited_bound |= stage2_ok(step)

        # -- scope: the queried window must cover the interval under investigation
        w = step.args.get("window")
        if w is None:
            pass                                   # defaults to the full horizon: never too narrow
        elif not (isinstance(w, (list, tuple)) and len(w) == 2):
            bad.append("malformed window")
        else:
            try:
                lo, hi = int(w[0]), int(w[1])
                if lo > fwin[0] or hi < fwin[1]:
                    bad.append(f"window {[lo, hi]} does not cover the focus interval "
                               f"{list(fwin)}")
            except (TypeError, ValueError):
                bad.append("non-integer window")
        if any("window" in b for b in bad) and mechanism != "compound":
            # On M4 a narrower window is not decomposition to be excused, nor scope error to be
            # flagged: it is the required behaviour, and it is judged by the stage-2 rule above.
            rep.scope_mismatch += 1

        # -- constraint: the bound the task states must be the bound the operator is given
        if mechanism == "cascade" and step.tool in ("temporal_reachability", "temporal_paths"):
            want = focus.get("max_wait")
            got = step.args.get("max_wait")
            if want is not None and got is None:
                rep.missing_constraint += 1
                bad.append(f"max_wait omitted (task states {want})")
            elif want is not None and got is not None and int(got) != int(want):
                rep.wrong_constraint += 1
                bad.append(f"max_wait={got} but the task states {want}")
        if mechanism == "compound" and step.tool in ("temporal_reachability", "temporal_paths"):
            want = focus.get("max_wait")
            got = step.args.get("max_wait")
            if want is not None and got is None:
                rep.missing_constraint += 1
                bad.append(f"max_wait omitted (task states {want})")
            elif want is not None and got is not None and int(got) != int(want):
                rep.wrong_constraint += 1
                bad.append(f"max_wait={got} but the task states {want}")
            # The compositional conjunct. Counted per call so the *rate* of unbound stage-2
            # queries stays visible, but phrased so it does not itself set cited_bad_support --
            # that verdict is made once, after the loop, over all cited calls together.
            if not stage2_ok(step):
                rep.missing_constraint += 1
                bad.append("unbound stage-2 window "
                           f"{step.args.get('window')} (admissible length "
                           f"{focus.get('admissible')})")
        if mechanism == "closure_wave" and step.tool == "temporal_closure":
            want, got = focus.get("delta"), step.args.get("delta")
            if want is not None and got is not None and int(got) != int(want):
                rep.wrong_constraint += 1
                bad.append(f"delta={got} but the task states {want}")
        if mechanism == "regime_switch" and step.tool == "neighborhood_evolution":
            want, got = focus.get("bandwidth"), step.args.get("bandwidth")
            if want is not None and got is not None and int(got) != int(want):
                rep.wrong_constraint += 1
                bad.append(f"bandwidth={got} but the task states {want}")

        # -- entities: the operator must be aimed at the nodes under investigation
        if mechanism == "cascade" and step.tool in ("temporal_reachability", "temporal_paths"):
            s, t = step.args.get("source"), step.args.get("target")
            if s is not None and t is not None:
                if {int(s), int(t)} != {int(focus["source"]), int(focus["target"])}:
                    rep.wrong_entities += 1
                    bad.append(f"aimed at {s}->{t}, task asks about "
                               f"{focus['source']}->{focus['target']}")
        if mechanism == "regime_switch" and step.tool == "neighborhood_evolution":
            g = step.args.get("group")
            if isinstance(g, list) and g and not set(map(int, g)) & set(focus.get("group", [])):
                rep.wrong_entities += 1
                bad.append("group does not intersect the focal group")

        if bad:
            rep.details.append(f"{step.step_id} {step.tool}: " + "; ".join(bad))
            asks_different_question = any("max_wait" in b or "delta" in b or "bandwidth" in b
                                          for b in bad)
            if asks_different_question and step.step_id in cited_support:
                rep.cited_bad_support = True

    if mechanism == "compound" and stage2_cited_any and not stage2_cited_bound:
        # Every cited stage-2 query ran over an interval wider than the admissible lag, so the
        # cited computation cannot distinguish "a cascade following the wave" from "a cascade
        # anywhere" -- which is true in both worlds. The conclusion may still be right; it is not
        # supported by what it cites.
        rep.cited_bad_support = True
    return rep


# ======================================================================================
# failure taxonomy (plan §18)
# ======================================================================================

CATEGORIES = ("static_shortcut", "recency_shortcut", "popularity_shortcut", "temporal_order_error",
              "window_selection_error", "tool_selection_error", "parameter_error",
              "evidence_mismatch", "code_failure", "budget_exhaustion", "malformed_output",
              "range_reconstruction", "correct")


def classify(trace: Trace, task: Task, mechanism: str, decision_correct: bool,
             evidence_grounded: bool) -> str:
    """Best-effort single label for what went wrong. Rule-based; no LLM judgement."""
    # Fabricated evidence is separated from ordinary schema failure because the two say opposite
    # things about a model. `malformed_output` reads as "could not produce the required JSON" --
    # a formatting deficiency, and easy to mistake for a defect in our harness. What the smaller
    # open-weight subjects actually do is emit *well-formed* JSON that cites step ids for tool
    # calls they never made: the agentic contract is broken, not the syntax. Naming it separately
    # keeps a fabrication from being reported as a formatting problem, and it belongs to the same
    # evidence-validity thread as `evidence_mismatch`.
    if not trace.steps and (trace.final_claim or trace.raw_final_response):
        cited = [e.get("trace_id") for e in (trace.final_claim or {}).get("evidence", [])]
        if cited or "trace_id" in (trace.raw_final_response or ""):
            return "fabricated_evidence"
    if trace.status == "malformed_output":
        return "malformed_output"
    if trace.status == "budget_exhausted":
        return "budget_exhaustion"
    if trace.status in ("tool_error", "api_error", "timeout"):
        return "code_failure"
    if decision_correct and evidence_grounded:
        return "correct"

    tools_used = Counter(s.tool for s in trace.steps)
    rep = parameterisation_report(trace, task, mechanism)

    if decision_correct and not evidence_grounded:
        return "evidence_mismatch"
    if rep.wrong_entities or rep.missing_constraint or rep.wrong_constraint:
        return "parameter_error"
    if rep.scope_mismatch:
        return "window_selection_error"
    if not (set(tools_used) & TEMPORAL_OPS):
        # no ordering-aware operator was used at all: did it try to reconstruct order instead?
        if detects_range_reconstruction(trace):
            return "range_reconstruction"
        if tools_used.get("top_active_nodes") or tools_used.get("node_degree"):
            return "popularity_shortcut"
        return "static_shortcut"
    return "temporal_order_error"


def detects_range_reconstruction(trace: Trace, min_calls: int = 3,
                                 min_windows: int = 2) -> bool:
    """Did the agent bisect time by repeated node-filtered range queries?

    The signature is several `window_event_count` calls that are node-filtered and pass multiple
    windows — the batched binary search observed in development. Detecting it lets the behaviour
    be reported as a measured strategy rather than inferred from narratives.
    """
    n = 0
    for s in trace.steps:
        if s.tool != "window_event_count" or not s.ok:
            continue
        wins = s.args.get("windows") or []
        if s.args.get("nodes") and isinstance(wins, list) and len(wins) >= min_windows:
            n += 1
    return n >= min_calls


def reconstruction_stats(trace: Trace) -> dict[str, Any]:
    """Cost signature of the range-reconstruction strategy: calls, windows per call, resolution."""
    calls, widths, per_call = 0, [], []
    for s in trace.steps:
        if s.tool != "window_event_count" or not s.ok:
            continue
        wins = s.args.get("windows") or []
        if not isinstance(wins, list) or not wins:
            continue
        calls += 1
        per_call.append(len(wins))
        for w in wins:
            try:
                widths.append(int(w[1]) - int(w[0]))
            except (TypeError, ValueError, IndexError):
                pass
    return {"range_calls": calls,
            "mean_windows_per_call": round(sum(per_call) / len(per_call), 2) if per_call else 0.0,
            "min_window_width": min(widths) if widths else None,
            "used_reconstruction": detects_range_reconstruction(trace)}
