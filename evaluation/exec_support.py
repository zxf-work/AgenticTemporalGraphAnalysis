"""Executed-support robustness variant of evidence validity (final readiness review §5).

EVPairAcc scores only the computation the agent *cites* as support, so a low value could in
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

Note that ExecEV is not mechanically >= EV: EV can hold via a cited static call (no temporal step
cited at all), while ExecEV demands an executed *deciding* operation with task-faithful arguments.
The two therefore decompose failures rather than bounding one another.

Applicability. The per-step test exists only where a surface exposes a *deciding* operator whose
arguments carry the task constraint: the A3 operators, and (by program inspection) A2. A0 and A1
expose no such operator, so ExecEV is **not defined** for them and `exec_ev` returns ``None``
rather than ``False``; a zero would read as a finding where no test was run. The same asymmetry
applies to the cited rule (`parameterisation_report`), which inspects only A3 operators and A2
programs: for A0/A1 the evidence criterion reduces to the structural conditions of
`evaluation/checker.py` (cited steps exist, cited windows lie inside the horizon, a positive
claim cites a support step), and EVPairAcc = PairAcc there is a property of what the checker can
see, not evidence that the cited support was task-faithful.
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


#: surfaces with no deciding operator: ExecEV is not applicable there (see module docstring)
NOT_APPLICABLE = ("A0", "A1")


def exec_ev(row: dict, trace, task, compound_tc: int | None = None) -> bool | None:
    """ExecEV_i for one scored run; ``None`` where the variant is not applicable (A0, A1)."""
    if row.get("condition") in NOT_APPLICABLE:
        return None
    correct = (row.get("decision") == "present") == (row.get("polarity") == "positive")
    claim_positive = (trace.final_claim or {}).get("decision") == "present"
    return correct and any(step_semvalid(s, task, row["mechanism"], row["condition"],
                                         compound_tc=compound_tc,
                                         claim_positive=claim_positive)
                           for s in trace.steps)
