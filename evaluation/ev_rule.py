"""Which evidence-validity rule is in force, and the switch that selects it.

Two rules exist, and every reported \\evpair{} value says which one produced it.

``stated`` (default) is the rule the paper defines (Eq. 3 and the trace-instrumentation
paragraph). A run is evidence-valid when

  * its decision is correct,
  * its claim cites no nonexistent step,
  * its support set -- the steps cited with role ``support`` that resolve to executed steps --
    is non-empty, and
  * **at least one** step in that set passes the per-step task-constraint test
    ``exec_support.step_semvalid`` (for A0 and A1, whose operations take no task-constraint
    argument, any successfully executed cited step passes).

``submitted`` is the rule the reviewed submission's numbers were computed with:
``check.evidence_grounded and not cited_bad_support``. It is existential for A2 and for the M4
stage-2 query, but for A3 on M1-M3 it fails a run as soon as *any* semantic call cited as support
carries a bound different from the task's, even when the task-bound call is cited beside it. It
also passes an *absent* claim that cites no support step, and fails a run that cites a window
outside the observable horizon. It is kept so the submitted numbers remain reproducible:

    python scripts/paper_numbers.py --ev-rule submitted
    ATGP_EV_RULE=submitted python scripts/figure1.py --data-only

On the 1,402-row corpus the two rules agree on every A0, A2 and M4 row; they differ on 58 A3 rows
of M1-M3 and on 2 A1 rows (README, "Evidence-validity rule"; asserted by scripts/test_ev_rule.py).
"""

from __future__ import annotations

import os

STATED, SUBMITTED = "stated", "submitted"
EV_RULES = (STATED, SUBMITTED)
ENV_VAR = "ATGP_EV_RULE"
FLAG = "--ev-rule"

_active: str | None = None


def _check(name: str) -> str:
    if name not in EV_RULES:
        raise ValueError(f"unknown evidence-validity rule {name!r}; choose one of {EV_RULES}")
    return name


def get_ev_rule() -> str:
    """The rule in force: an explicit `set_ev_rule`, else $ATGP_EV_RULE, else ``stated``."""
    if _active is not None:
        return _active
    return _check(os.environ.get(ENV_VAR, STATED))


def set_ev_rule(name: str) -> str:
    global _active
    _active = _check(name)
    return _active


def ev_rule_from_argv(argv: list[str]) -> str:
    """Honour ``--ev-rule NAME`` / ``--ev-rule=NAME`` on a script's command line.

    Removes the flag from ``argv`` in place (several scripts treat every remaining non-flag
    argument as a run directory) and returns the rule now in force.
    """
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == FLAG:
            if i + 1 >= len(argv):
                raise SystemExit(f"{FLAG} needs a value: one of {EV_RULES}")
            set_ev_rule(argv[i + 1])
            del argv[i:i + 2]
            continue
        if a.startswith(FLAG + "="):
            set_ev_rule(a.split("=", 1)[1])
            del argv[i]
            continue
        i += 1
    return get_ev_rule()
