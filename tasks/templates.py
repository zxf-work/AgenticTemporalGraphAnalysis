"""Task wording (plan §11).

Two hand-written paraphrases per mechanism, chosen deterministically by seed parity (R8). No
runtime LLM generation: the wording must be identical across agent conditions and reproducible
from the frozen config, and a generated question would be an uncontrolled variable and a
contamination surface.

Wording rules
-------------
* never reveal polarity — the same sentence introduces a positive and a negative world;
* never name the planted structure or its parameters beyond what ``focus`` legitimately scopes;
* state the decision options and the output contract explicitly, identically for every condition;
* explicitly warn against the shortcut the confounders are designed to bait, so that a failure is
  a failure of analysis rather than of instruction-following.
"""

from __future__ import annotations

DECISION_BLOCK = """
Return your conclusion as a JSON object with exactly these fields:
  "task_id":    the task id given above
  "decision":   "present" | "absent" | "insufficient"
  "mechanism":  "{mechanism}"
  "entities":   the node ids the conclusion concerns (list of integers)
  "window":     [start, end] — the interval your conclusion concerns
  "confidence": a number in [0, 1]
  "evidence":   a list of {{"trace_id": "step_NN", "role": "support" | "confounder_control"}}
                citing the tool results you actually relied on. "support" marks a result that
                supports your decision; "confounder_control" marks a result that tests an
                alternative explanation.
  "narrative":  a short free-text explanation (optional, not scored)

Cite only step ids that appear in this session. Do not invent results.
""".strip()

CAUTION = ("Base your conclusion on the temporal evidence itself. The total volume of activity, "
           "the recent popularity of the nodes involved, and the existence of a connection when "
           "time is ignored are each insufficient on their own.")

# --------------------------------------------------------------------------------------
# M1 cascade
# --------------------------------------------------------------------------------------

M1 = [
    # paraphrase 1 — direct analytical wording
    """Task {task_id}.

In this event graph, each record is an interaction (u, v, t) meaning u contacted v at time t.

Investigate whether the activity involving source node {source} and target node {target} during
the interval [{w0}, {w1}] supports a genuine time-respecting cascade: a sequence of interactions
that could have carried influence from {source} to {target}, where each successive interaction
occurs after the previous one and follows it by no more than {max_wait} time units.

{caution}

{decision_block}""",

    # paraphrase 2 — incident-investigation wording
    """Task {task_id}.

You are reviewing an incident log. Each record (u, v, t) means account u interacted with account
v at time t.

An analyst suspects that something propagated from account {source} to account {target} between
time {w0} and time {w1}. Propagation is only possible along interactions that happen in order,
with no more than {max_wait} time units of delay between one step and the next. Determine whether
such a propagation route actually exists in this log.

{caution}

{decision_block}""",
]

# --------------------------------------------------------------------------------------
# M2 closure wave
# --------------------------------------------------------------------------------------

M2 = [
    """Task {task_id}.

In this event graph, each record is an interaction (u, v, t) meaning u contacted v at time t.

Among the nodes {nodes_preview}, determine whether the interval [{w0}, {w1}] contains a genuine
temporal closure wave: pairs that share a common contact going on to interact with each other
within {delta} time units of that shared contact being established. A closure wave requires the
closing interactions to follow their supporting interactions promptly, not merely to exist
somewhere in the log.

{caution}

{decision_block}""",

    """Task {task_id}.

You are studying how a community knits together. Each record (u, v, t) means member u interacted
with member v at time t.

For the members {nodes_preview}, decide whether the period [{w0}, {w1}] shows real triadic
closure dynamics: after two members become connected through a shared contact, do they then
connect to each other within {delta} time units? Accumulated triangles that formed in no
particular order do not constitute a closure wave.

{caution}

{decision_block}""",
]

# --------------------------------------------------------------------------------------
# M3 regime switch
# --------------------------------------------------------------------------------------

M3 = [
    """Task {task_id}.

In this event graph, each record is an interaction (u, v, t) meaning u contacted v at time t.

The group of nodes {nodes_preview} is under review. Determine whether this group underwent an
abrupt change of interaction partners during [{w0}, {w1}] — an old set of contacts giving way to
a new set within a short transition — rather than gradual background turnover spread across the
period. Compare the group's contacts before and after a candidate change point using a half-window
of {bandwidth} time units on each side.

{caution}

{decision_block}""",

    """Task {task_id}.

You are auditing a set of accounts. Each record (u, v, t) means account u interacted with account
v at time t.

For the accounts {nodes_preview}, decide whether their circle of contacts was replaced abruptly at
some point in [{w0}, {w1}], or whether their contacts drifted gradually over the whole period.
Compare the contacts in the {bandwidth} time units before a candidate change point against the
{bandwidth} time units after it. Both patterns can end with the same final set of contacts, so the
distinction is about timing.

{caution}

{decision_block}""",
]

# --------------------------------------------------------------------------------------
# M4 compound investigation
# --------------------------------------------------------------------------------------
#
# States both stages and, explicitly, the dependency between them. The dependency has to be
# stated: it is the thing under test, and an unstated one would make this a guessing game about
# our intent rather than a test of compositional analysis. What is NOT stated is t_c -- deriving
# the second stage's window from the first stage's result is the competence being measured.

M4 = [
    """Task {task_id}.

In this event graph, each record is an interaction (u, v, t) meaning u contacted v at time t.

This is a two-part investigation about the interval [{w0}, {w1}], and both parts must hold.

First: determine whether the group {nodes_preview} shows a temporal closure wave — pairs sharing a
common contact going on to interact with each other within {delta} time units of that shared
contact being established. Note when that wave finishes, that is, the time of the last such
closing interaction.

Second: determine whether influence could then have travelled from node {source} to node {target}
along interactions that happen in order, with no more than {max_wait} time units between one step
and the next — and, critically, whether that route begins no more than {admissible} time units
after the closure wave finished.

Report the mechanism as present only if BOTH parts hold AND the route falls in the admissible
period following the wave. A closure wave and a route that are each present but unrelated in time
do NOT satisfy this.

{caution}

{decision_block}""",

    """Task {task_id}.

You are reconstructing an incident. Each record (u, v, t) means account u interacted with account
v at time t.

Analysts believe a coordination phase preceded a propagation phase within [{w0}, {w1}].

The coordination phase is a closure wave among the accounts {nodes_preview}: accounts sharing a
contact then connecting to each other within {delta} time units. Establish whether it occurred and
when the last such connection was made.

The propagation phase is a route from account {source} to account {target} along interactions in
increasing time order with gaps of at most {max_wait}. For the incident to be confirmed, this route
must start within {admissible} time units of the moment the coordination phase ended.

Confirm the incident only if both phases occurred and the propagation began inside that admissible
period. Two phases that happened at unrelated times are not this incident.

{caution}

{decision_block}""",
]

TEMPLATES = {"cascade": M1, "closure_wave": M2, "regime_switch": M3, "compound": M4}


def render(mechanism: str, paraphrase: int, *, task_id: str, focus: dict,
           window: tuple[int, int]) -> str:
    """Render the served question. ``paraphrase`` is 1-based."""
    tpl = TEMPLATES[mechanism][(paraphrase - 1) % len(TEMPLATES[mechanism])]
    nodes = focus.get("nodes") or focus.get("group") or []
    preview = ", ".join(str(n) for n in nodes[:12]) + (" ..." if len(nodes) > 12 else "")
    return tpl.format(
        task_id=task_id, source=focus.get("source"), target=focus.get("target"),
        max_wait=focus.get("max_wait"), delta=focus.get("delta"),
        bandwidth=focus.get("bandwidth"), admissible=focus.get("admissible"),
        nodes_preview=f"[{preview}]", w0=window[0], w1=window[1], caution=CAUTION,
        decision_block=DECISION_BLOCK.format(mechanism=mechanism),
    ).strip()
