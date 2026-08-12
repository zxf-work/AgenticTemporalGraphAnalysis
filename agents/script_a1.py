"""Script-A1-Bisect — a deterministic reconstruction algorithm on the A1 interface (T16).

Answers the question the LLM result alone cannot: is a frontier model *efficiently* exploiting the
information available through adaptive range queries, or merely finding an expensive route to it?
Without this reference point the paper's cost curves describe one model's idiosyncrasy rather than
the interface's query complexity.

The algorithm uses **only A1 tools**, through the same dispatch and trace machinery as the LLM
agents, so its calls are counted the same way:

1. read the static structure around the endpoints (`khop_subgraph`);
2. enumerate candidate source->target routes in that structure — pure computation on results
   already returned, costing no extra calls;
3. for each candidate route, localise each edge's timestamp by batched binary search
   (`edges_between` over shrinking windows) to a resolution fine enough to compare against the
   waiting bound;
4. stop as soon as one route is shown time-respecting, or when all routes are refuted.

Reported: calls to decision, and whether the decision was right. Cost is dominated by
`(number of candidate routes) x (route length) x log_W(R)`, which is exactly the prediction the
complexity grid tests.
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agents.tools_spec import surface_for
from common import stable_hash
from tasks.schema import Task
from traces.schema import Step, Trace, make_run_id
from worlds.base import TemporalWorld


@dataclass
class ScriptResult:
    decision: str
    calls: int
    trace: Trace
    localised: dict[tuple[int, int], tuple[int, int]] = field(default_factory=dict)
    candidates: int = 0
    note: str = ""


class _Dispatcher:
    """Runs A1 tools, recording each call in a trace exactly as the LLM harness would."""

    def __init__(self, world: TemporalWorld, trace: Trace, budget: int):
        self.world = world
        self.trace = trace
        self.budget = budget
        self.tools = {t.name: t for t in surface_for("A1")}

    @property
    def exhausted(self) -> bool:
        return len(self.trace.steps) >= self.budget

    def __call__(self, name: str, **args: Any) -> Any:
        if self.exhausted:
            raise BudgetExhausted()
        result = self.tools[name].fn(self.world, **args)
        self.trace.steps.append(Step(
            step_id=self.trace.next_step_id(), tool=name, args=dict(args),
            result_digest=stable_hash(result), result_summary=str(result)[:400],
            latency_ms=0, tokens_before=0, tokens_after=0))
        return result


class BudgetExhausted(RuntimeError):
    pass


def run_script_a1(world: TemporalWorld, task: Task, *, tool_budget: int = 20,
                  windows_per_call: int = 8, max_candidates: int = 8,
                  world_hash: str = "", pair_id: str = "") -> ScriptResult:
    """Deterministic cascade decision using only the adaptive aggregate-range interface."""
    focus = task.focus
    source, target = int(focus["source"]), int(focus["target"])
    lo, hi = int(focus["window"][0]), int(focus["window"][1])
    max_wait = int(focus["max_wait"])

    trace = Trace(run_id=make_run_id(task.task_id, "A1", "script-a1-bisect", "static",
                                     world_hash or world.world_hash()),
                  task_id=task.task_id, pair_id=pair_id,
                  world_hash=world_hash or world.world_hash(), agent_condition="A1",
                  model="script-a1-bisect", prompt_hash="static", code_commit="",
                  tool_budget=tool_budget)
    call = _Dispatcher(world, trace, tool_budget)

    def claim(decision: str, cited: list[str]) -> dict:
        """The same claim shape the agents must produce.

        The reference previously set only ``{"decision": ...}``, and on one path set nothing at
        all, so its verdicts scored as ``none`` and its pair accuracy read 0.00 regardless of what
        the algorithm decided. A reference that is scored differently from the subjects is not a
        reference. Emitting the full contract also makes EVPairAcc meaningful for it.
        """
        return {"task_id": task.task_id, "decision": decision, "mechanism": "cascade",
                "entities": [source, target], "window": [lo, hi], "confidence": 1.0,
                "evidence": [{"trace_id": s, "role": "support"} for s in cited],
                "narrative": "Script-A1-Bisect deterministic reference."}

    try:
        sub = call("khop_subgraph", node=source, k=6, window=[lo, hi])
        adj: dict[int, list[int]] = {}
        for u, v in sub["edges"]:
            adj.setdefault(u, []).append(v)
        routes = _enumerate_routes(adj, source, target, max_len=7, limit=max_candidates)
        if not routes:
            # This path decided "absent" on real evidence and must record it. Returning without
            # setting final_claim scored the verdict as `none`, which is how a working algorithm
            # came to report 0.00 pair accuracy on every cell.
            trace.final_claim = claim("absent", [s.step_id for s in trace.steps])
            return ScriptResult("absent", len(trace.steps), trace, candidates=0,
                                note="no static route between the endpoints")

        localised: dict[tuple[int, int], tuple[int, int]] = {}
        for route in routes:
            edges = list(zip(route, route[1:]))
            bounds = []
            feasible = True
            for (u, v) in edges:
                if (u, v) in localised:
                    bounds.append(localised[(u, v)])
                    continue
                b = _localise(call, u, v, lo, hi, max_wait, windows_per_call)
                if b is None:
                    feasible = False
                    break
                localised[(u, v)] = b
                bounds.append(b)
            if not feasible:
                continue
            if _time_respecting(bounds, max_wait):
                trace.final_claim = claim("present", [s.step_id for s in trace.steps])
                return ScriptResult("present", len(trace.steps), trace, localised=localised,
                                    candidates=len(routes),
                                    note=f"route {route} is time-respecting")
        trace.final_claim = claim("absent", [s.step_id for s in trace.steps])
        return ScriptResult("absent", len(trace.steps), trace, localised=localised,
                            candidates=len(routes),
                            note=f"all {len(routes)} candidate routes refuted")

    except BudgetExhausted:
        trace.status = "budget_exhausted"
        return ScriptResult("insufficient", len(trace.steps), trace,
                            note="budget exhausted before a decision")


# ---------------------------------------------------------------------------- helpers

def _enumerate_routes(adj: dict[int, list[int]], source: int, target: int, *, max_len: int,
                      limit: int) -> list[list[int]]:
    """Simple directed routes source->target in the already-retrieved static structure."""
    out: list[list[int]] = []

    def dfs(node: int, path: list[int], seen: set[int]) -> None:
        if len(out) >= limit or len(path) > max_len:
            return
        for nxt in sorted(adj.get(node, ())):
            if nxt in seen:
                continue
            if nxt == target:
                out.append(path + [nxt])
                if len(out) >= limit:
                    return
                continue
            dfs(nxt, path + [nxt], seen | {nxt})

    dfs(source, [source], {source})
    return sorted(out, key=len)


def _localise(call: _Dispatcher, u: int, v: int, lo: int, hi: int, max_wait: int,
              windows_per_call: int) -> tuple[int, int] | None:
    """Narrow the time of the u->v interaction to a bracket finer than half the waiting bound.

    Uses the interface's *batched* form: ``window_event_count`` accepts a list of windows, so one
    call returns the whole occupancy profile of a bracket split ``windows_per_call`` ways. The
    cost is therefore ``log_W(R)`` calls per edge, not ``log_2(R)`` and emphatically not one call
    per window tested — the naive per-window form is what made an early version of this reference
    algorithm 7x more expensive than the LLM it was meant to calibrate.

    ``edges_between`` disambiguates only at the final bracket, where the node's activity may
    include decoy edges as well as the one under test.
    """
    # Resolution must be fine relative to the waiting bound, not merely comparable to it: the
    # feasibility test below is conservative over the whole bracket, so brackets as wide as
    # max_wait/2 refute genuinely feasible routes.
    target_width = max(1, max_wait // 8)

    def search(a: int, b: int, depth: int = 0) -> tuple[int, int] | None:
        if depth > 6:
            return None
        if (b - a) <= target_width:
            conf = call("edges_between", u=u, v=v, window=[a, b])
            return (a, b) if conf["n_events_u_to_v"] > 0 else None
        width = max(1, (b - a) // windows_per_call)
        wins = []
        for i in range(windows_per_call):
            w0 = a + i * width
            if w0 > b:
                break
            wins.append([w0, min(b, w0 + width - 1)])
        if not wins:
            return None
        prof = call("window_event_count", windows=wins, nodes=[u, v])
        # Every occupied bracket must be tried, not just the first: the node filter counts decoy
        # activity through u as well, so the earliest occupied bracket need not be the one holding
        # the u->v interaction.
        for c in prof["counts"]:
            if c["n_events"] > 0:
                got = search(int(c["window"][0]), int(c["window"][1]), depth + 1)
                if got is not None:
                    return got
        return None

    return search(lo, hi)


def _time_respecting(bounds: list[tuple[int, int]], max_wait: int) -> bool:
    """Decide feasibility from localisation brackets, conservatively.

    A route counts as time-respecting only when the ordering and the waiting bound hold for the
    whole bracket, so an imprecise localisation can never produce a false positive.
    """
    for (a0, a1), (b0, b1) in zip(bounds, bounds[1:]):
        if not b0 > a1:                       # must be strictly later even in the worst case
            return False
        if (b1 - a0) > max_wait:              # gap must fit the bound even at its widest
            return False
    return True
