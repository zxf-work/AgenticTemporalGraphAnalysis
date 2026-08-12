"""Tool surfaces exposed to each agent condition.

A tool is a name, a description, a JSON-schema parameter block, and a Python callable bound to a
loaded world. The *same* dispatch machinery serves every condition; conditions differ only in
which tools appear in their surface (plan §13). That is what makes the A1/A3 comparison a
statement about the analytical interface rather than about harness engineering.

Descriptions are written to be neutral: they say what the operator computes, never when to use it
and never anything about mechanisms, polarity, or the benchmark.
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass
from typing import Any, Callable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools import static_surface as S
from tools import temporal_surface as T
from tools.code_surface import ALLOWED_IMPORTS, run_python
from worlds.base import TemporalWorld


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]          # JSON schema
    fn: Callable[..., Any]
    ordering_aware: bool = False        # True => excluded from the A1 surface


def _obj(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required}


_WINDOW = {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2,
           "description": "[start, end] inclusive time interval"}
_NODES = {"type": "array", "items": {"type": "integer"}, "description": "node ids"}

# ======================================================================================
# static / windowed surface — available to every condition
# ======================================================================================

STATIC_TOOLS: list[ToolSpec] = [
    ToolSpec("graph_summary",
             "Event count, distinct edge count, active node count and repeat ratio for a time "
             "window.",
             _obj({"window": _WINDOW}, []),
             lambda w, window=None: S.graph_summary(w, window=_w(window))),

    ToolSpec("node_degree",
             "Per-node event count and distinct-partner count within a time window.",
             _obj({"nodes": _NODES, "window": _WINDOW}, ["nodes"]),
             lambda w, nodes, window=None: S.node_degree(w, nodes, window=_w(window))),

    ToolSpec("neighbors",
             "Distinct neighbours of a node within a time window. Returns identities only, not "
             "the times of the interactions.",
             _obj({"node": {"type": "integer"}, "window": _WINDOW,
                   "direction": {"type": "string", "enum": ["in", "out", "both"]}}, ["node"]),
             lambda w, node, window=None, direction="both":
                 S.neighbors(w, node, window=_w(window), direction=direction)),

    ToolSpec("khop_subgraph",
             "Nodes and edges within k hops of a centre node, ignoring event order.",
             _obj({"node": {"type": "integer"}, "k": {"type": "integer"}, "window": _WINDOW},
                  ["node"]),
             lambda w, node, k=2, window=None: S.khop_subgraph(w, node, k=k, window=_w(window))),

    ToolSpec("triangle_count",
             "Number of static triangles incident to the given nodes within a window.",
             _obj({"nodes": _NODES, "window": _WINDOW}, []),
             lambda w, nodes=None, window=None: S.triangle_count(w, nodes, window=_w(window))),

    ToolSpec("top_active_nodes",
             "The k nodes with the most events in a window.",
             _obj({"k": {"type": "integer"}, "window": _WINDOW}, []),
             lambda w, k=20, window=None: S.top_active_nodes(w, k=k, window=_w(window))),

    ToolSpec("window_event_count",
             "Event counts for a list of time windows, optionally restricted to given nodes.",
             _obj({"windows": {"type": "array", "items": _WINDOW}, "nodes": _NODES}, ["windows"]),
             lambda w, windows, nodes=None: S.window_event_count(w, windows, nodes=nodes)),

    ToolSpec("edges_between",
             "How many interactions occurred between two nodes in a window, per direction.",
             _obj({"u": {"type": "integer"}, "v": {"type": "integer"}, "window": _WINDOW},
                  ["u", "v"]),
             lambda w, u, v, window=None: S.edges_between(w, u, v, window=_w(window))),

    ToolSpec("compute",
             "Evaluate a small arithmetic expression over named values.",
             _obj({"expression": {"type": "string"},
                   "values": {"type": "object", "additionalProperties": {"type": "number"}}},
                  ["expression"]),
             lambda w, expression, values=None: T.compute(expression, values)),
]

# ======================================================================================
# ordering-aware surface — A3 and A4 only
# ======================================================================================

TEMPORAL_TOOLS: list[ToolSpec] = [
    ToolSpec("temporal_reachability",
             "Whether a target can be reached from a source along interactions that occur in "
             "increasing time order, optionally requiring that consecutive interactions are "
             "separated by no more than max_wait. Returns the earliest-arrival route when one "
             "exists.",
             _obj({"source": {"type": "integer"}, "target": {"type": "integer"},
                   "window": _WINDOW, "max_wait": {"type": "integer"}}, ["source"]),
             lambda w, source, target=None, window=None, max_wait=None:
                 _reach(w, source, target, window, max_wait),
             ordering_aware=True),

    ToolSpec("temporal_paths",
             "Up to max_paths distinct time-respecting routes between two nodes.",
             _obj({"source": {"type": "integer"}, "target": {"type": "integer"},
                   "window": _WINDOW, "max_wait": {"type": "integer"},
                   "max_paths": {"type": "integer"}}, ["source", "target"]),
             lambda w, source, target, window=None, max_wait=None, max_paths=3:
                 {"paths": T.temporal_paths(w, source, target, window=_w(window),
                                            max_wait=max_wait, max_paths=max_paths)},
             ordering_aware=True),

    ToolSpec("temporal_closure",
             "For triangles among the given nodes, the fraction whose third edge arrived within "
             "delta time units of the second — that is, triangles that closed promptly rather "
             "than at an arbitrary time. Also reports the median closing lag.",
             _obj({"nodes": _NODES, "window": _WINDOW, "delta": {"type": "integer"}}, []),
             lambda w, nodes=None, window=None, delta=800:
                 _closure(w, nodes, window, delta),
             ordering_aware=True),

    ToolSpec("neighborhood_evolution",
             "Series of neighbourhood turnover for a group of nodes: at each probe time, one "
             "minus the Jaccard overlap between the group's contacts before and after that time.",
             _obj({"group": _NODES, "window": _WINDOW, "bandwidth": {"type": "integer"}},
                  ["group"]),
             lambda w, group, window=None, bandwidth=5000:
                 _regime(w, group, window, bandwidth),
             ordering_aware=True),

    ToolSpec("graph_metric_timeseries",
             "A metric binned over time: event_count, unique_edges or degree.",
             _obj({"metric": {"type": "string",
                              "enum": ["event_count", "unique_edges", "degree"]},
                   "window": _WINDOW, "n_bins": {"type": "integer"}, "nodes": _NODES}, []),
             lambda w, metric="event_count", window=None, n_bins=50, nodes=None:
                 T.graph_metric_timeseries(w, metric, window=_w(window), n_bins=n_bins,
                                           nodes=nodes),
             ordering_aware=True),

    ToolSpec("burst_detection",
             "Time bins whose activity exceeds a z-score threshold above the mean.",
             _obj({"window": _WINDOW, "n_bins": {"type": "integer"},
                   "z_threshold": {"type": "number"}, "nodes": _NODES}, []),
             lambda w, window=None, n_bins=50, z_threshold=2.0, nodes=None:
                 T.burst_detection(w, window=_w(window), n_bins=n_bins,
                                   z_threshold=z_threshold, nodes=nodes),
             ordering_aware=True),

    ToolSpec("diff_snapshots",
             "Edges added and removed between two time windows.",
             _obj({"window_a": _WINDOW, "window_b": _WINDOW, "nodes": _NODES},
                  ["window_a", "window_b"]),
             lambda w, window_a, window_b, nodes=None:
                 T.diff_snapshots(w, tuple(window_a), tuple(window_b), nodes=nodes),
             ordering_aware=True),

    ToolSpec("co_active",
             "How often the given nodes are active in the same time bins.",
             _obj({"nodes": _NODES, "window": _WINDOW, "n_bins": {"type": "integer"}}, ["nodes"]),
             lambda w, nodes, window=None, n_bins=40:
                 T.co_active(w, nodes, window=_w(window), n_bins=n_bins),
             ordering_aware=True),
]


# -------------------------------------------------------------------- result shaping
# Bounded, agent-readable summaries. Full results are hashed into the trace; what the model sees
# is capped so that context pressure is identical across conditions.

def _w(window: Any) -> tuple[int, int] | None:
    return (int(window[0]), int(window[1])) if window else None


def _reach(w: TemporalWorld, source: int, target: int | None, window: Any,
           max_wait: int | None) -> dict[str, Any]:
    r = T.temporal_reachability(w, source, target, window=_w(window), max_wait=max_wait)
    out: dict[str, Any] = {"source": source, "target": target,
                           "window": list(_w(window) or w.horizon), "max_wait": max_wait,
                           "n_nodes_reached": r.n_reached}
    if target is not None:
        out["reached"] = r.reached
        if r.reached:
            out["route"] = [{"from": u, "to": v, "t": t} for u, v, t in r.path]
            out["hop_waits"] = r.hop_waits
            out["max_wait_used"] = r.max_wait_used
    return out


def _closure(w: TemporalWorld, nodes: Any, window: Any, delta: int) -> dict[str, Any]:
    c = T.temporal_closure(w, nodes, window=_w(window), delta=delta)
    return {"n_triangles": c.n_triangles, "n_closed_within_delta": c.n_closed_within_delta,
            "fraction_closed": round(c.fraction, 4), "delta": delta,
            "median_closing_lag": round(c.median_closing_lag, 1),
            "examples": [{"nodes": [a, b, cc], "t_wedge_complete": tw, "t_close": tc}
                         for a, b, cc, tw, tc in c.closures[:15]]}


def _regime(w: TemporalWorld, group: Any, window: Any, bandwidth: int) -> dict[str, Any]:
    r = T.neighborhood_evolution(w, group, window=_w(window), bandwidth=bandwidth)
    return {"peak_change": round(r.peak_change, 4), "peak_time": r.peak_time,
            "bandwidth": bandwidth,
            "series": [{"t": t, "change": round(v, 4)} for t, v in r.series],
            "contacts_before_peak": r.before[:40], "contacts_after_peak": r.after[:40]}


# ======================================================================================
# A0 — order-oblivious control surface
# ======================================================================================
#
# Every A0 tool is *fixed to the focus horizon*: no tool accepts a window, so no sequence of A0
# calls can narrow attention to a sub-interval. This is what makes the A0 acceptance condition
# decidable rather than sampled — the query space is (tools x node arguments), which
# `scripts/test_a0_equivalence.py` enumerates exhaustively over every node.
#
# The distinction A0 draws is the point of the pivot: A1's tools are individually non-ordering,
# yet collectively order-revealing under adaptive composition, because they take a window. Remove
# the window and the interface is order-oblivious in the strong sense: its entire observable
# surface is identical on the two members of a matched pair.

A0_TOOLS: list[ToolSpec] = [
    ToolSpec("graph_summary_full",
             "Event count, distinct edge count, active node count and repeat ratio over the "
             "whole observation horizon.",
             _obj({}, []),
             lambda w: S.graph_summary(w, window=None)),

    ToolSpec("node_degree_full",
             "Per-node event count and distinct-partner count over the whole horizon.",
             _obj({"nodes": _NODES}, ["nodes"]),
             lambda w, nodes: S.node_degree(w, nodes, window=None)),

    ToolSpec("neighbors_full",
             "Distinct neighbours of a node over the whole horizon.",
             _obj({"node": {"type": "integer"},
                   "direction": {"type": "string", "enum": ["in", "out", "both"]}}, ["node"]),
             lambda w, node, direction="both": S.neighbors(w, node, window=None,
                                                           direction=direction)),

    ToolSpec("khop_subgraph_full",
             "Nodes and edges within k hops of a centre node over the whole horizon.",
             _obj({"node": {"type": "integer"}, "k": {"type": "integer"}}, ["node"]),
             lambda w, node, k=2: S.khop_subgraph(w, node, k=k, window=None)),

    ToolSpec("triangle_count_full",
             "Static triangles incident to the given nodes over the whole horizon.",
             _obj({"nodes": _NODES}, []),
             lambda w, nodes=None: S.triangle_count(w, nodes, window=None)),

    ToolSpec("top_active_nodes_full",
             "The k nodes with the most events over the whole horizon.",
             _obj({"k": {"type": "integer"}}, []),
             lambda w, k=20: S.top_active_nodes(w, k=k, window=None)),

    ToolSpec("edges_between_full",
             "How many interactions occurred between two nodes over the whole horizon.",
             _obj({"u": {"type": "integer"}, "v": {"type": "integer"}}, ["u", "v"]),
             lambda w, u, v: S.edges_between(w, u, v, window=None)),

    ToolSpec("compute",
             "Evaluate a small arithmetic expression over named values.",
             _obj({"expression": {"type": "string"},
                   "values": {"type": "object", "additionalProperties": {"type": "number"}}},
                  ["expression"]),
             lambda w, expression, values=None: T.compute(expression, values)),
]

ALL_TOOLS = {t.name: t for t in A0_TOOLS + STATIC_TOOLS + TEMPORAL_TOOLS}


# ======================================================================================
# A2 programmable surface (milestone C3)
# ======================================================================================
#
# One tool, deliberately. A2's burden is to *synthesise an algorithm*, so giving it a menu of
# helpers alongside the interpreter would blur it back toward A1/A3 and make the comparison
# unreadable. The description states the environment precisely and, as everywhere else, says
# nothing about mechanisms, polarity, or when to use it.

CODE_TOOLS: list[ToolSpec] = [
    ToolSpec("run_python",
             "Execute Python 3 against the full event log and return anything it prints. "
             "The variable `events` is a list of (u, v, t) tuples sorted by t, where u "
             "interacted with v at time t; `horizon` is the (start, end) of the observable "
             "period. State is NOT preserved between calls -- each call runs in a fresh "
             "interpreter, so every call must be self-contained. Print what you want to see; "
             "nothing is returned automatically. Available imports: "
             + ", ".join(ALLOWED_IMPORTS) + ".",
             _obj({"code": {"type": "string",
                            "description": "Python source to execute. Use print() for output."}},
                  ["code"]),
             lambda world, code: run_python(world, code),
             ordering_aware=True),
]


def surface_for(condition: str) -> list[ToolSpec]:
    """Interface hierarchy (pivot addendum §3).

    * ``A0`` order-oblivious control — full-horizon aggregates only, no window parameter
    * ``A1`` adaptive aggregate-range interface — windowed counts, no ordering semantics
    * ``A2`` raw-event / code interface — sandbox instead of tools
    * ``A3``/``A4`` semantic temporal operators, on top of A1's surface
    """
    if condition == "A0":
        return list(A0_TOOLS)
    if condition == "A1":
        return list(STATIC_TOOLS)
    if condition in ("A3", "A4"):
        return list(STATIC_TOOLS) + list(TEMPORAL_TOOLS)
    if condition == "A2":
        # A sandbox *is* its surface: raw events plus an interpreter, and no temporal operators.
        return list(CODE_TOOLS)
    raise ValueError(f"unknown condition {condition!r}")
