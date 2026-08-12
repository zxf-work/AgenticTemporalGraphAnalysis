"""Static and windowed graph operators — the A1 baseline's tool surface (plan §13-A1).

A1 is meant to be a *capable* graph agent that simply lacks ordering-aware primitives, so this
surface is deliberately generous: degrees, neighbourhoods, k-hop subgraphs, triangle counts,
summaries, activity rankings, and windowed counts. What it does not contain is any operator that
respects event order — no time-respecting reachability, no δ-closure, no neighbourhood-change
series.

Fairness note carried into the paper: because windows are a parameter, a determined agent could in
principle localise individual events in time by repeated narrow-window queries and reconstruct
ordering by hand within its tool budget. We do not block that. Whether agents actually do it is an
empirical finding, and ``evaluation/error_taxonomy.py`` detects the pattern (many narrow windowed
calls on the same node pair) so it can be reported rather than assumed away.
"""

from __future__ import annotations

import pathlib
import sys
from collections import Counter, defaultdict
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from worlds.base import TemporalWorld


def _mask(world: TemporalWorld, window: tuple[int, int] | None) -> np.ndarray:
    lo, hi = window if window is not None else world.horizon
    return world.window_mask(lo, hi)


def node_degree(world: TemporalWorld, nodes: Sequence[int], *,
                window: tuple[int, int] | None = None) -> dict[str, Any]:
    """Event count and distinct-partner count per node, optionally within a window."""
    m = _mask(world, window)
    src, dst = world.src[m].tolist(), world.dst[m].tolist()
    ev = Counter()
    partners: dict[int, set[int]] = defaultdict(set)
    for u, v in zip(src, dst):
        ev[u] += 1
        ev[v] += 1
        partners[u].add(v)
        partners[v].add(u)
    return {"window": list(window or world.horizon),
            "nodes": [{"node": int(n), "events": int(ev.get(n, 0)),
                       "distinct_partners": len(partners.get(n, ()))} for n in nodes]}


def neighbors(world: TemporalWorld, node: int, *, window: tuple[int, int] | None = None,
              direction: str = "both", limit: int = 200) -> dict[str, Any]:
    """Distinct neighbours of a node within a window (no timing information)."""
    m = _mask(world, window)
    out, inn = set(), set()
    for u, v in zip(world.src[m].tolist(), world.dst[m].tolist()):
        if u == node:
            out.add(v)
        if v == node:
            inn.add(u)
    sel = {"out": out, "in": inn, "both": out | inn}[direction]
    return {"node": node, "window": list(window or world.horizon), "direction": direction,
            "n_neighbors": len(sel), "neighbors": sorted(sel)[:limit]}


def khop_subgraph(world: TemporalWorld, node: int, k: int = 2, *,
                  window: tuple[int, int] | None = None, max_nodes: int = 500) -> dict[str, Any]:
    """Node set and edge list of the k-hop neighbourhood, ignoring event order.

    Expansion and truncation are both by node id, never by set-iteration order. Otherwise the
    result would depend on the order events happen to appear in the table — which differs between
    two worlds that carry the same edges at re-assigned times, and would hand an agent a
    difference that has no analytical meaning.
    """
    m = _mask(world, window)
    adj: dict[int, set[int]] = defaultdict(set)
    for u, v in zip(world.src[m].tolist(), world.dst[m].tolist()):
        adj[u].add(v)
        adj[v].add(u)

    seen = {node}
    frontier = [node]
    for _ in range(k):
        nxt: set[int] = set()
        for x in frontier:
            nxt |= adj.get(x, set())
        frontier = sorted(nxt - seen)
        seen |= nxt
    kept = sorted(seen)[:max_nodes]
    kept_set = set(kept)
    edges = sorted({(u, v) for u in kept for v in adj.get(u, ()) if v in kept_set})
    return {"center": node, "k": k, "n_nodes": len(kept), "n_edges": len(edges),
            "nodes": kept, "edges": edges[:1000], "truncated": len(seen) > max_nodes}


def triangle_count(world: TemporalWorld, nodes: Sequence[int] | None = None, *,
                   window: tuple[int, int] | None = None) -> dict[str, Any]:
    """Static triangle count — identical for both members of a matched pair, by construction."""
    m = _mask(world, window)
    adj: dict[int, set[int]] = defaultdict(set)
    for u, v in zip(world.src[m].tolist(), world.dst[m].tolist()):
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    focus = sorted(set(nodes)) if nodes else sorted(adj)
    total, per_node = 0, {}
    for u in focus:
        c = 0
        nb = sorted(adj.get(u, ()))
        for i in range(len(nb)):
            for j in range(i + 1, len(nb)):
                if nb[j] in adj.get(nb[i], ()):
                    c += 1
        per_node[u] = c
        total += c
    return {"window": list(window or world.horizon), "n_triangles_incident": total,
            "per_node": dict(sorted(per_node.items())[:100])}


def graph_summary(world: TemporalWorld, *, window: tuple[int, int] | None = None) -> dict[str, Any]:
    m = _mask(world, window)
    src, dst, t = world.src[m], world.dst[m], world.t[m]
    edges = set(zip(src.tolist(), dst.tolist()))
    active = set(src.tolist()) | set(dst.tolist())
    return {"window": list(window or world.horizon), "n_events": int(m.sum()),
            "n_distinct_edges": len(edges), "n_active_nodes": len(active),
            "n_nodes_total": world.n_nodes,
            "t_min": int(t.min()) if m.any() else None,
            "t_max": int(t.max()) if m.any() else None,
            "repeat_ratio": round(1 - len(edges) / max(1, int(m.sum())), 4)}


def top_active_nodes(world: TemporalWorld, k: int = 20, *,
                     window: tuple[int, int] | None = None) -> dict[str, Any]:
    """The k busiest nodes. Ties break by node id, never by insertion order."""
    m = _mask(world, window)
    c = Counter(world.src[m].tolist()) + Counter(world.dst[m].tolist())
    ranked = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    return {"window": list(window or world.horizon),
            "top": [{"node": int(n), "events": int(v)} for n, v in ranked]}


def window_event_count(world: TemporalWorld, windows: Sequence[Sequence[int]], *,
                       nodes: Sequence[int] | None = None) -> dict[str, Any]:
    """Event counts in each of several windows — the static agent's view of 'when'."""
    out = []
    for w in windows:
        lo, hi = int(w[0]), int(w[1])
        m = world.window_mask(lo, hi)
        if nodes:
            ns = list(nodes)
            m = m & (np.isin(world.src, ns) | np.isin(world.dst, ns))
        out.append({"window": [lo, hi], "n_events": int(m.sum())})
    return {"counts": out}


def edges_between(world: TemporalWorld, u: int, v: int, *,
                  window: tuple[int, int] | None = None) -> dict[str, Any]:
    """How many times u and v interacted in a window, in either direction. No timestamps."""
    m = _mask(world, window)
    su, sv = world.src[m], world.dst[m]
    fwd = int(((su == u) & (sv == v)).sum())
    bwd = int(((su == v) & (sv == u)).sum())
    return {"u": u, "v": v, "window": list(window or world.horizon),
            "n_events_u_to_v": fwd, "n_events_v_to_u": bwd, "connected": bool(fwd + bwd)}
