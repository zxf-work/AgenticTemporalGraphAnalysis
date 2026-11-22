r"""Executed-support robustness variant of evidence validity (final readiness review §5).

\evpair{} scores only the computation the agent *cites* as support, so a low value could in
principle reflect citation diligence rather than the absence of a task-faithful computation: an
agent might execute a correctly bound operation and then cite a broad exploratory call instead.
ExecEV closes that loophole by re-running the same per-step task-constraint test over ALL executed
steps in the trace, cited or not:

    ExecEV_i       = correct_i AND (exists executed step s: SemValid(s, q_i) = 1)
    ExecEVPairAcc  = share of matched pairs with ExecEV on both members

The per-step test mirrors `error_taxonomy.parameterisation_report` exactly -- the same argument
comparisons that set `cited_bad_support` when they fail on a cited step -- restricted to the
operator that decides each mechanism (the restriction the cited rule inherits implicitly from the
citation itself). The citation-related conditions (citation exists, support role present, scope of
cited windows) are deliberately dropped; that is the point of the variant.

This module also holds the cited-support form of the same test, which is the evidence-validity
rule the paper states (`cited_ev`; see `evaluation/ev_rule.py` for the rule switch):

    EV_i = correct_i AND no cited step is nonexistent
                     AND (exists s in S_i^support: SemValid(s, q_i) = 1)

Under that rule ExecEV >= EV on A2 and A3 by construction, because the support step that passes is
itself an executed step. Under the `submitted` rule the two are not ordered: there EV can hold via
a cited static call (no temporal step cited at all), while ExecEV demands an executed *deciding*
operation with task-faithful arguments.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from evaluation.error_taxonomy import (CONSTRAINT_KEY, _compound_stage2_anchored,
                                       _compound_stage2_bound)

#: the operator whose result decides each mechanism's question
DECIDING_OPS = {"cascade": ("temporal_reachability", "temporal_paths"),
                "compound": ("temporal_reachability", "temporal_paths"),
                "closure_wave": ("temporal_closure",),
                "regime_switch": ("neighborhood_evolution",)}


#: interfaces whose operations take no task-constraint argument
NO_CONSTRAINT_INTERFACES = ("A0", "A1")


def _int_eq(got, want) -> bool:
    try:
        return int(got) == int(want)
    except (TypeError, ValueError):
        return False


def step_semvalid(step, task, mechanism: str, condition: str,
                  compound_tc: int | None = None, claim_positive: bool = False) -> bool:
    """Would this executed step pass the task-constraint check had it been the sole support?

    ``compound_tc``/``claim_positive`` select the anchored claim-sensitive M4 rule, exactly as in
    `parameterisation_report`; without ``compound_tc`` the width-only proxy applies.
    """
    if not step.ok:
        return False
    if condition in NO_CONSTRAINT_INTERFACES:
        # A0 and A1 operations take no task-constraint argument (no waiting bound, closure bound
        # or bandwidth can be passed to them), so there is nothing for the test to compare: any
        # successfully executed step satisfies it.
        return True
    focus = task.focus
    if condition == "A2":
        # mirror of `_a2_constraint_report`: the program must reference the stated bound
        if step.tool != "run_python":
            return False
        key = CONSTRAINT_KEY.get(mechanism)
        want = focus.get(key) if key else None
        if want is None:
            return True
        code = str(step.args.get("code", ""))
        return (str(int(want)) in code) or (key in code)
    if step.tool not in DECIDING_OPS.get(mechanism, ()):
        return False
    if mechanism in ("cascade", "compound"):
        want, got = focus.get("max_wait"), step.args.get("max_wait")
        if want is not None and (got is None or not _int_eq(got, want)):
            return False
        if mechanism == "compound":
            if compound_tc is not None:
                return _compound_stage2_anchored(step, focus, compound_tc, claim_positive)
            return _compound_stage2_bound(step, focus)
        return True
    if mechanism == "closure_wave":
        want, got = focus.get("delta"), step.args.get("delta")
        return not (want is not None and got is not None and not _int_eq(got, want))
    if mechanism == "regime_switch":
        want, got = focus.get("bandwidth"), step.args.get("bandwidth")
        return not (want is not None and got is not None and not _int_eq(got, want))
    return False


def exec_ev(row: dict, trace, task, compound_tc: int | None = None) -> bool:
    """ExecEV_i for one scored run."""
    correct = (row.get("decision") == "present") == (row.get("polarity") == "positive")
    claim_positive = (trace.final_claim or {}).get("decision") == "present"
    return correct and any(step_semvalid(s, task, row["mechanism"], row["condition"],
                                         compound_tc=compound_tc,
                                         claim_positive=claim_positive)
                           for s in trace.steps)


# ======================================================================================
# the cited-support rule the paper states (evidence validity, Eq. 3)
# ======================================================================================

def cited_support_steps(trace) -> tuple[list, list[str]]:
    """(S_i^support, nonexistent ids): the steps cited with role ``support`` that resolve to
    executed steps, and every cited id -- of either role -- that resolves to none."""
    by_id = {s.step_id: s for s in trace.steps}
    evidence = (trace.final_claim or {}).get("evidence", []) or []
    support = [by_id[e["trace_id"]] for e in evidence
               if e.get("role") == "support" and e.get("trace_id") in by_id]
    nonexistent = [str(e.get("trace_id")) for e in evidence if e.get("trace_id") not in by_id]
    return support, nonexistent


def support_semvalid(trace, task, mechanism: str, condition: str,
                     compound_tc: int | None = None) -> bool:
    """Does at least one cited support step pass the per-step task-constraint test?

    The existential half of the stated rule. False for an empty support set, so an *absent* claim
    that cites no support step is not evidence-valid.
    """
    claim_positive = (trace.final_claim or {}).get("decision") == "present"
    support, _ = cited_support_steps(trace)
    return any(step_semvalid(s, task, mechanism, condition, compound_tc=compound_tc,
                             claim_positive=claim_positive) for s in support)


def cited_ev(trace, task, mechanism: str, condition: str, gold: bool,
             compound_tc: int | None = None) -> bool:
    """EV_i under the rule the paper states, for one run.

    ``gold`` is True when the mechanism is present in this world. An abstention or a missing
    claim is incorrect. A claim citing any nonexistent step is evidence-invalid whatever else it
    cites; exploratory calls cited beside a step that passes are not penalised.
    """
    decision = (trace.final_claim or {}).get("decision")
    if decision != ("present" if gold else "absent"):
        return False
    _, nonexistent = cited_support_steps(trace)
    if nonexistent:
        return False
    return support_semvalid(trace, task, mechanism, condition, compound_tc=compound_tc)
