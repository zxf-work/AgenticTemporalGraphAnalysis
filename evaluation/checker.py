"""Claim checking against the execution trace (taskboard B5).

Correctness here is mechanical. A decision is *evidence-grounded correct* only when the decision
matches gold **and** the citations survive inspection: every cited step exists, the required
evidence roles are present, and no cited step reached outside the task's temporal scope.

That last check is this paper's small piece of temporal faithfulness: an agent may not justify a
conclusion about window W with a tool call that looked outside W.
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from evaluation.claim_schema import Claim
from traces.schema import Trace


@dataclass
class CheckResult:
    decision_correct: bool
    abstained: bool
    citations_exist: bool
    has_support: bool
    has_confounder_control: bool
    scope_respected: bool
    evidence_grounded: bool
    unsupported_positive: bool
    reasons: list[str]


def check(trace: Trace, gold: bool, *, condition: str, horizon: tuple[int, int],
          scope: tuple[int, int] | None = None) -> CheckResult:
    """Score one run. ``gold`` is True when the mechanism is genuinely present."""
    reasons: list[str] = []
    claim_d = trace.final_claim
    if claim_d is None:
        return CheckResult(False, False, False, False, False, False, False, False,
                           [f"no claim ({trace.status})"])

    claim = Claim.from_dict(claim_d)

    # R3: gold is binary, so an abstention is scored as incorrect and tracked separately.
    abstained = claim.is_abstention
    decision_correct = (claim.decision == "present") == gold and not abstained
    if abstained:
        reasons.append("abstained")
    elif not decision_correct:
        reasons.append(f"decision {claim.decision} vs gold {'present' if gold else 'absent'}")

    step_ids = trace.step_ids()
    cited = [e.trace_id for e in claim.evidence]
    missing = [c for c in cited if c not in step_ids]
    citations_exist = not missing
    if missing:
        reasons.append(f"cited nonexistent steps {missing}")

    has_support = bool(claim.cited("support"))
    has_control = bool(claim.cited("confounder_control"))

    scope_respected = True
    lo, hi = scope if scope else horizon
    for e in claim.evidence:
        step = trace.step(e.trace_id)
        if step is None:
            continue
        w = step.args.get("window") or step.args.get("window_a")
        if isinstance(w, (list, tuple)) and len(w) == 2:
            try:
                if int(w[0]) < horizon[0] or int(w[1]) > horizon[1]:
                    scope_respected = False
                    reasons.append(f"{e.trace_id} queried outside the observable horizon")
            except (TypeError, ValueError):
                pass

    # A4 additionally requires a control citation for a positive claim (plan §13-A4).
    needs_control = (condition == "A4" and claim.is_positive)
    evidence_grounded = (decision_correct and citations_exist and scope_respected
                         and (has_support or not claim.is_positive)
                         and (has_control or not needs_control))
    if decision_correct and not evidence_grounded:
        reasons.append("correct decision but evidence does not support it")

    unsupported_positive = claim.is_positive and not (
        citations_exist and has_support and (has_control or condition != "A4"))

    return CheckResult(decision_correct=decision_correct, abstained=abstained,
                       citations_exist=citations_exist, has_support=has_support,
                       has_confounder_control=has_control, scope_respected=scope_respected,
                       evidence_grounded=evidence_grounded,
                       unsupported_positive=unsupported_positive, reasons=reasons)


def localization(claim_d: dict[str, Any], truth_entities: list[int],
                 truth_window: tuple[int, int]) -> dict[str, float]:
    """Entity F1 and window IoU — scored on positive worlds only (plan §16.5)."""
    claim = Claim.from_dict(claim_d)
    pred, truth = set(claim.entities), set(truth_entities)
    tp = len(pred & truth)
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(truth) if truth else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    a0, a1 = claim.window
    b0, b1 = truth_window
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return {"entity_precision": prec, "entity_recall": rec, "entity_f1": f1,
            "window_iou": (inter / union) if union else 0.0}
