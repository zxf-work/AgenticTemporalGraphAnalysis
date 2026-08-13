"""Executed-support robustness variant of evidence validity (final readiness review §5).

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

Note that ExecEV is not mechanically >= EV: EV can hold via a cited static call (no temporal step
cited at all), while ExecEV demands an executed *deciding* operation with task-faithful arguments.
The two therefore decompose failures rather than bounding one another.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from evaluation.error_taxonomy import CONSTRAINT_KEY, _compound_stage2_bound

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


def step_semvalid(step, task, mechanism: str, condition: str) -> bool:
    """Would this executed step pass the task-constraint check had it been the sole support?"""
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
        if mechanism == "compound" and not _compound_stage2_bound(step, focus):
            return False
        return True
    if mechanism == "closure_wave":
        want, got = focus.get("delta"), step.args.get("delta")
        return not (want is not None and got is not None and not _int_eq(got, want))
    if mechanism == "regime_switch":
        want, got = focus.get("bandwidth"), step.args.get("bandwidth")
        return not (want is not None and got is not None and not _int_eq(got, want))
    return False


def exec_ev(row: dict, trace, task) -> bool:
    """ExecEV_i for one scored run."""
    correct = (row.get("decision") == "present") == (row.get("polarity") == "positive")
    return correct and any(step_semvalid(s, task, row["mechanism"], row["condition"])
                           for s in trace.steps)
