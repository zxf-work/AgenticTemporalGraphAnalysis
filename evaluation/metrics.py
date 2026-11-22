"""Aggregate metrics (plan §16), with pair accuracy as the headline.

Chance floors (R4), stated in the paper so a single-world number is never read as success:

* a deterministic order-blind agent scores **exactly 0** pair accuracy — the static surface is
  identical across a matched pair, so it must answer both members the same way, and exactly one of
  those answers is wrong;
* independent 50/50 guessing scores **25%**;
* always-present and always-absent each score **0**.
"""

from __future__ import annotations

import pathlib
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from evaluation.checker import CheckResult
from evaluation.ev_rule import SUBMITTED, get_ev_rule

CONSISTENCY = ("correct_flip", "same_positive", "same_negative", "reverse_flip", "incomplete")


@dataclass
class RunRow:
    """One scored agent run."""
    pair_id: str
    polarity: str
    gold: bool
    condition: str
    model: str
    mechanism: str
    confounder: str
    decision: str
    check: CheckResult
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    status: str = "success"
    tool_protocol_errors: int = 0        # malformed tool calls (never an analytical failure)
    # C6 backend cost, kept strictly apart from the interaction cost above
    rows_scanned: int = 0                # events the data system examined
    backend_ms: int = 0                  # time inside operator implementations
    llm_calls: int = 0                   # LLM rounds (>= tool_calls)
    # interface-relative observability instrumentation (pivot addendum §7, §13)
    param_error_rate: float = 0.0        # share of temporal calls mis-parameterised
    cited_bad_support: bool = False      # a conclusion rests on a different-question call
    decomposition_rate: float = 0.0      # sub-interval / waypoint probes (not an error)
    n_temporal_calls: int = 0
    used_reconstruction: bool = False    # bisected time via node-filtered range queries
    range_calls: int = 0
    windows_per_call: float = 0.0
    min_window_width: int | None = None
    failure_category: str = ""

    # set by `score_run.score`: at least one step cited as support, and resolving to an executed
    # step, passes the per-step task-constraint test (`exec_support.support_semvalid`)
    support_semvalid: bool = False

    # ---- C2: evidence-valid analytical correctness -------------------------------------
    # EV_i under the rule the paper states (Eq. 3; `evaluation/ev_rule.py`):
    #
    #   EV_i = [answer correct] AND [no cited step is nonexistent]
    #          AND [at least one cited support step satisfies SemValid]
    #
    # All conjuncts are deterministic, per the milestone's requirement that no LLM judge acts as
    # the primary validity oracle where a parameter check is possible. The rule is existential
    # over cited support on every interface and mechanism: an exploratory call cited beside the
    # task-bound computation does not disqualify a run. An empty support set is evidence-invalid,
    # for *absent* claims too.
    #
    # Legitimate decomposition (narrower windows, intermediate waypoints) is deliberately NOT a
    # disqualifier; it is reported separately as `decomposition_rate`.
    @property
    def ev_valid(self) -> bool:
        if get_ev_rule() == SUBMITTED:
            return self.ev_valid_submitted
        return bool(self.check.decision_correct and self.check.citations_exist
                    and self.support_semvalid)

    # The rule the reviewed submission's numbers were computed with, kept so they stay
    # reproducible (`--ev-rule submitted`):
    #
    #   * `check.evidence_grounded` requires the decision to be right, the cited steps to EXIST,
    #     the citations to stay inside the observable horizon, and a positive claim to cite
    #     support;
    #   * `cited_bad_support` fails a run whose cited support asked a *different question*. For A2
    #     and for the M4 stage-2 query it fires only when NO cited support call is task-bound;
    #     for A3 on M1-M3 it fires as soon as ANY cited support call carries a different bound,
    #     which is stricter than the rule the paper states.
    @property
    def ev_valid_submitted(self) -> bool:
        return bool(self.check.evidence_grounded) and not self.cited_bad_support


@dataclass
class Summary:
    n_runs: int = 0
    n_pairs: int = 0
    single_world_acc: float = 0.0
    pair_acc: float = 0.0
    negative_fpr: float = 0.0
    abstention_rate: float = 0.0
    evidence_grounded_acc: float = 0.0
    ev_pair_acc: float = 0.0                 # C2: EVPairAcc, the paired form of ev_valid
    ev_gap: float = 0.0                      # pair_acc - ev_pair_acc
    unsupported_positive_rate: float = 0.0
    consistency: dict[str, float] = field(default_factory=dict)
    mean_tool_calls: float = 0.0
    mean_input_tokens: float = 0.0
    mean_output_tokens: float = 0.0
    mean_latency_s: float = 0.0
    # A3 parameterisation (reported metric, not a rebuttal)
    param_error_rate: float = 0.0            # mean share of temporal calls mis-parameterised
    cited_bad_support_rate: float = 0.0      # share of runs whose support cites a bad call
    # A1 range-reconstruction behaviour
    reconstruction_rate: float = 0.0
    mean_range_calls: float = 0.0
    mean_windows_per_call: float = 0.0
    failure_mix: dict[str, float] = field(default_factory=dict)
    pair_ids_correct: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self.__dict__.items() if k != "pair_ids_correct"}


def summarize(rows: Iterable[RunRow]) -> Summary:
    rows = list(rows)
    s = Summary(n_runs=len(rows))
    if not rows:
        return s

    s.single_world_acc = sum(r.check.decision_correct for r in rows) / len(rows)
    s.abstention_rate = sum(r.check.abstained for r in rows) / len(rows)

    negatives = [r for r in rows if not r.gold]
    s.negative_fpr = (sum(r.decision == "present" for r in negatives) / len(negatives)
                      if negatives else 0.0)

    positives_claimed = [r for r in rows if r.decision == "present"]
    s.unsupported_positive_rate = (
        sum(r.check.unsupported_positive for r in positives_claimed) / len(positives_claimed)
        if positives_claimed else 0.0)

    s.evidence_grounded_acc = sum(r.check.evidence_grounded for r in rows) / len(rows)

    # -------------------------------------------------- paired analysis
    by_pair: dict[str, dict[str, RunRow]] = defaultdict(dict)
    for r in rows:
        by_pair[r.pair_id][r.polarity] = r

    counts = {k: 0 for k in CONSISTENCY}
    n_complete = 0
    n_ev = 0
    for pid, d in by_pair.items():
        pos, neg = d.get("positive"), d.get("negative")
        if pos is None or neg is None:
            counts["incomplete"] += 1
            continue
        n_complete += 1
        p_ok = pos.decision == "present"
        n_ok = neg.decision == "absent"
        if p_ok and n_ok:
            counts["correct_flip"] += 1
            s.pair_ids_correct.append(pid)
            # EVPairAcc = (1/N) sum EV_i^+ EV_i^-: a pair counts only when BOTH worlds were
            # decided correctly AND on evidence that addressed the question asked.
            if pos.ev_valid and neg.ev_valid:
                n_ev += 1
        elif pos.decision == "present" and neg.decision == "present":
            counts["same_positive"] += 1
        elif pos.decision == "absent" and neg.decision == "absent":
            counts["same_negative"] += 1
        elif pos.decision == "absent" and neg.decision == "present":
            counts["reverse_flip"] += 1
        else:
            counts["incomplete"] += 1        # an abstention on one side

    s.n_pairs = n_complete
    s.pair_acc = (counts["correct_flip"] / n_complete) if n_complete else 0.0
    s.ev_pair_acc = (n_ev / n_complete) if n_complete else 0.0
    s.ev_gap = s.pair_acc - s.ev_pair_acc
    denom = max(1, n_complete)
    s.consistency = {k: round(v / denom, 4) for k, v in counts.items()}

    s.mean_tool_calls = sum(r.tool_calls for r in rows) / len(rows)
    s.mean_input_tokens = sum(r.input_tokens for r in rows) / len(rows)
    s.mean_output_tokens = sum(r.output_tokens for r in rows) / len(rows)
    s.mean_latency_s = sum(r.latency_ms for r in rows) / len(rows) / 1000.0

    # parameterisation is only meaningful for runs that used a temporal operator at all
    with_temporal = [r for r in rows if r.n_temporal_calls > 0]
    s.param_error_rate = (sum(r.param_error_rate for r in with_temporal) / len(with_temporal)
                          if with_temporal else 0.0)
    s.cited_bad_support_rate = (sum(r.cited_bad_support for r in with_temporal)
                                / len(with_temporal) if with_temporal else 0.0)

    s.reconstruction_rate = sum(r.used_reconstruction for r in rows) / len(rows)
    s.mean_range_calls = sum(r.range_calls for r in rows) / len(rows)
    recon = [r for r in rows if r.range_calls]
    s.mean_windows_per_call = (sum(r.windows_per_call for r in recon) / len(recon)
                               if recon else 0.0)

    cats = Counter(r.failure_category for r in rows if r.failure_category)
    s.failure_mix = {k: round(v / len(rows), 4) for k, v in cats.most_common()}
    return s


def by_group(rows: Iterable[RunRow], key: str) -> dict[str, Summary]:
    """Stratify (``condition``, ``mechanism``, ``confounder``, ``model``)."""
    groups: dict[str, list[RunRow]] = defaultdict(list)
    for r in rows:
        groups[getattr(r, key)].append(r)
    return {k: summarize(v) for k, v in sorted(groups.items())}


# ======================================================================================
# chance floors — asserted in the self-test so the paper can state them as facts
# ======================================================================================

CHANCE_FLOORS = {
    "always_present": 0.0,
    "always_absent": 0.0,
    "coin_flip": 0.25,
    "static_oracle": 0.0,
}


def expected_floor(policy: str) -> float:
    return CHANCE_FLOORS[policy]
