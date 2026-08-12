"""Ordering-aware temporal graph operators.

Written from scratch for this study (DECISIONS D2). These functions serve two roles:

*  **gold oracles** — the generator and the invariant checker call them to label worlds;
*  **agent tools** — conditions A3/A4 expose them through the tool interface.

That dual role is deliberate and safe: the operators compute *primitives* (reachability, closure
fractions, neighbourhood change), never the task's answer. The agent still has to choose which
primitive to invoke, over which nodes, and in which window, and then interpret the result. No
function here takes a "is the mechanism present?" form, and none reads generator metadata.
"""

from __future__ import annotations

import math
import pathlib
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from worlds.base import TemporalWorld

INF = float("inf")


# ======================================================================================
# time-respecting reachability
# ======================================================================================

@dataclass
class ReachResult:
    reached: bool
    arrival: dict[int, int]              # node -> earliest arrival time
    path: list[tuple[int, int, int]]     # [(u, v, t)] for the earliest-arrival path, if reached
    hop_waits: list[int]                 # idle time before each hop, source departure excluded
    n_reached: int

    @property
    def max_wait_used(self) -> int:
        return max(self.hop_waits) if self.hop_waits else 0


def temporal_reachability(world: TemporalWorld, source: int, target: int | None = None, *,
                          window: tuple[int, int] | None = None,
                          max_wait: int | None = None) -> ReachResult:
    """Earliest-arrival time-respecting reachability.

    An edge ``(u, v, t)`` may be traversed when ``arrival[u] <= t`` and, if ``max_wait`` is set,
    ``t - arrival[u] <= max_wait`` — the walker may not idle at an *intermediate* node longer than
    Δ. Departure from the source is unconstrained: the source is available throughout the window,
    so a padded observation window never makes a genuine cascade infeasible.

    A single forward sweep in time order is exact for earliest arrival: any path reaching ``u``
    uses only events at times ``<= t``, all of which have already been processed.
    """
    lo, hi = window if window is not None else world.horizon
    order = np.lexsort((world.dst, world.src, world.t))
    src, dst, ts = world.src[order], world.dst[order], world.t[order]
    mask = (ts >= lo) & (ts <= hi)
    src, dst, ts = src[mask].tolist(), dst[mask].tolist(), ts[mask].tolist()

    arrival: dict[int, int] = {source: lo}
    pred: dict[int, tuple[int, int]] = {}

    def _can_depart(node: int, at: int, now: int) -> bool:
        if now < at:
            return False
        if max_wait is None or node == source:
            return True
        return (now - at) <= max_wait

    for u, v, t in zip(src, dst, ts):
        au = arrival.get(u)
        if au is not None and _can_depart(u, au, t):
            if v not in arrival or arrival[v] > t:
                arrival[v] = t
                pred[v] = (u, t)
        if not world.directed:
            av = arrival.get(v)
            if av is not None and _can_depart(v, av, t):
                if u not in arrival or arrival[u] > t:
                    arrival[u] = t
                    pred[u] = (v, t)

    path: list[tuple[int, int, int]] = []
    hop_waits: list[int] = []
    reached = target is not None and target in arrival
    if reached:
        node = target
        while node != source and node in pred:
            u, t = pred[node]
            path.append((u, node, t))
            if u != source:
                hop_waits.append(int(t - arrival[u]))
            node = u
        path.reverse()
        hop_waits.reverse()
    return ReachResult(reached=reached, arrival=arrival, path=path, hop_waits=hop_waits,
                       n_reached=len(arrival))


def temporal_paths(world: TemporalWorld, source: int, target: int, *,
                   window: tuple[int, int] | None = None, max_wait: int | None = None,
                   max_paths: int = 3, max_depth: int = 8) -> list[list[tuple[int, int, int]]]:
    """Up to ``max_paths`` distinct time-respecting paths, found by bounded DFS in time order."""
    lo, hi = window if window is not None else world.horizon
    out_ev: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for u, v, t in zip(world.src.tolist(), world.dst.tolist(), world.t.tolist()):
        if lo <= t <= hi:
            out_ev[u].append((t, v))
            if not world.directed:
                out_ev[v].append((t, u))
    for lst in out_ev.values():
        lst.sort()

    found: list[list[tuple[int, int, int]]] = []

    def dfs(node: int, at: int, acc: list[tuple[int, int, int]], seen: set[int]) -> None:
        if len(found) >= max_paths or len(acc) >= max_depth:
            return
        # The waiting bound applies between consecutive hops, not to departure from the source
        # (D7). This must match temporal_reachability exactly: two operators that disagree about
        # feasibility would make an agent's answer depend on which one it happened to call.
        at_source = not acc
        for t, nxt in out_ev.get(node, ()):
            if t < at:
                continue
            if max_wait is not None and not at_source and (t - at) > max_wait:
                continue
            if nxt in seen:
                continue
            step = acc + [(node, nxt, t)]
            if nxt == target:
                found.append(step)
                if len(found) >= max_paths:
                    return
                continue
            dfs(nxt, t, step, seen | {nxt})

    dfs(source, lo, [], {source})
    return found


# ======================================================================================
# temporal closure (triadic) motifs
# ======================================================================================

@dataclass
class ClosureResult:
    n_triangles: int
    n_closed_within_delta: int
    fraction: float
    closures: list[tuple[int, int, int, int, int]]   # (a, b, c, t_wedge, t_close)
    median_closing_lag: float


def temporal_closure(world: TemporalWorld, nodes: Sequence[int] | None = None, *,
                     window: tuple[int, int] | None = None, delta: int = 800,
                     max_triangles: int = 20000) -> ClosureResult:
    """δ-temporal triadic closure, measured per triangle.

    For each triangle among the given nodes, take the first occurrence time of its three edges and
    sort them ``t1 <= t2 <= t3``. The triangle *closed in time* when ``t3 - t2 <= delta``: the
    third edge arrived promptly after the wedge was complete, rather than at an arbitrary moment.
    The reported fraction is over triangles, which is the quantity the mechanism is defined on, so
    the agent's tool and the gold oracle compute exactly the same thing.

    The static triangle set is identical between a matched pair — only ``t3 - t2`` differs — so a
    static motif counter returns the same value for both worlds.
    """
    lo, hi = window if window is not None else world.horizon
    focus = set(nodes) if nodes else None

    first: dict[tuple[int, int], int] = {}
    adj: dict[int, set[int]] = defaultdict(set)
    for u, v, t in zip(world.src.tolist(), world.dst.tolist(), world.t.tolist()):
        if not (lo <= t <= hi) or u == v:
            continue
        a, b = (u, v) if u <= v else (v, u)
        if (a, b) not in first or first[(a, b)] > t:
            first[(a, b)] = t
        adj[u].add(v)
        adj[v].add(u)

    triangles: list[tuple[int, int, int]] = []
    centres = sorted(focus) if focus else sorted(adj)
    seen: set[tuple[int, int, int]] = set()
    for a in centres:
        nbrs = sorted(x for x in adj.get(a, ()) if x != a)
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                b, c = nbrs[i], nbrs[j]
                if c not in adj.get(b, ()):
                    continue
                key = tuple(sorted((a, b, c)))          # type: ignore[assignment]
                if key in seen:
                    continue
                seen.add(key)                            # type: ignore[arg-type]
                triangles.append(key)                    # type: ignore[arg-type]
                if len(triangles) >= max_triangles:
                    break

    closures, lags = [], []
    for a, b, c in triangles:
        ts = [first.get((min(x, y), max(x, y))) for x, y in ((a, b), (b, c), (a, c))]
        if any(t is None for t in ts):
            continue
        t1, t2, t3 = sorted(ts)                          # type: ignore[misc]
        lags.append(t3 - t2)
        if (t3 - t2) <= delta:
            closures.append((a, b, c, t2, t3))

    n = len(triangles)
    frac = (len(closures) / n) if n else 0.0
    median_lag = float(np.median(lags)) if lags else 0.0
    return ClosureResult(n_triangles=n, n_closed_within_delta=len(closures), fraction=frac,
                         closures=closures[:200], median_closing_lag=median_lag)


# ======================================================================================
# neighbourhood evolution / regime change
# ======================================================================================

@dataclass
class RegimeResult:
    peak_change: float
    peak_time: int
    series: list[tuple[int, float]]      # (t, 1 - Jaccard)
    before: list[int]
    after: list[int]


def neighborhood_evolution(world: TemporalWorld, group: Sequence[int], *,
                           window: tuple[int, int] | None = None, bandwidth: int = 5000,
                           n_points: int = 40) -> RegimeResult:
    """Windowed neighbourhood turnover for a focal group.

    At each probe time ``t`` compare the neighbour set of ``group`` in ``[t - bandwidth, t)`` with
    that in ``[t, t + bandwidth)`` and report ``1 - Jaccard``. An abrupt regime switch produces a
    sharp peak; the same edges spread out gradually produce a flat, low curve.
    """
    lo, hi = window if window is not None else world.horizon
    gset = set(group)
    ev: list[tuple[int, int]] = []       # (t, partner)
    for u, v, t in zip(world.src.tolist(), world.dst.tolist(), world.t.tolist()):
        if not (lo <= t <= hi):
            continue
        if u in gset:
            ev.append((t, v))
        if v in gset:
            ev.append((t, u))
    ev.sort()
    times = [t for t, _ in ev]

    series: list[tuple[int, float]] = []
    best = (-1.0, lo, set(), set())
    if not ev:
        return RegimeResult(0.0, lo, [], [], [])

    probes = np.linspace(lo + bandwidth, hi - bandwidth, max(n_points, 2))
    for p in probes:
        p = int(p)
        before = {q for t, q in ev if p - bandwidth <= t < p}
        after = {q for t, q in ev if p <= t < p + bandwidth}
        if not before and not after:
            series.append((p, 0.0))
            continue
        inter = len(before & after)
        union = len(before | after)
        val = 1.0 - (inter / union if union else 1.0)
        series.append((p, val))
        if val > best[0] and before and after:
            best = (val, p, before, after)

    return RegimeResult(peak_change=max(0.0, best[0]), peak_time=best[1], series=series,
                        before=sorted(best[2]), after=sorted(best[3]))


# ======================================================================================
# generic descriptive operators (available to A3/A4; static analogues live in static_surface)
# ======================================================================================

def graph_metric_timeseries(world: TemporalWorld, metric: str = "event_count", *,
                            window: tuple[int, int] | None = None, n_bins: int = 50,
                            nodes: Sequence[int] | None = None) -> dict[str, Any]:
    """Coarse activity/structure series over time. metric ∈ {event_count, unique_edges, degree}."""
    lo, hi = window if window is not None else world.horizon
    m = world.window_mask(lo, hi)
    t = world.t[m]
    src, dst = world.src[m], world.dst[m]
    if nodes:
        keep = np.isin(src, list(nodes)) | np.isin(dst, list(nodes))
        t, src, dst = t[keep], src[keep], dst[keep]
    edges = np.linspace(lo, hi + 1, n_bins + 1).astype(np.int64)
    out = []
    for i in range(n_bins):
        b = (t >= edges[i]) & (t < edges[i + 1])
        if metric == "event_count":
            val = int(b.sum())
        elif metric == "unique_edges":
            val = len(set(zip(src[b].tolist(), dst[b].tolist())))
        elif metric == "degree":
            val = len(set(src[b].tolist()) | set(dst[b].tolist()))
        else:
            raise ValueError(f"unknown metric {metric!r}")
        out.append({"t_start": int(edges[i]), "t_end": int(edges[i + 1]) - 1, "value": val})
    return {"metric": metric, "window": [lo, hi], "bins": out}


def burst_detection(world: TemporalWorld, *, window: tuple[int, int] | None = None,
                    n_bins: int = 50, z_threshold: float = 2.0,
                    nodes: Sequence[int] | None = None) -> dict[str, Any]:
    """Bins whose activity exceeds ``z_threshold`` standard deviations above the mean."""
    series = graph_metric_timeseries(world, "event_count", window=window, n_bins=n_bins,
                                     nodes=nodes)
    vals = np.array([b["value"] for b in series["bins"]], dtype=float)
    mu, sd = float(vals.mean()), float(vals.std())
    bursts = []
    for b, v in zip(series["bins"], vals):
        z = (v - mu) / sd if sd > 0 else 0.0
        if z >= z_threshold:
            bursts.append({**b, "z": round(z, 2)})
    return {"mean": round(mu, 2), "sd": round(sd, 2), "z_threshold": z_threshold,
            "n_bursts": len(bursts), "bursts": bursts}


def diff_snapshots(world: TemporalWorld, window_a: tuple[int, int],
                   window_b: tuple[int, int], nodes: Sequence[int] | None = None) -> dict[str, Any]:
    """Edges present in B but not A, and vice versa, optionally restricted to ``nodes``."""
    def edge_set(w: tuple[int, int]) -> set[tuple[int, int]]:
        m = world.window_mask(*w)
        pairs = set(zip(world.src[m].tolist(), world.dst[m].tolist()))
        if nodes:
            ns = set(nodes)
            pairs = {(u, v) for u, v in pairs if u in ns or v in ns}
        return pairs

    a, b = edge_set(window_a), edge_set(window_b)
    added, removed = sorted(b - a), sorted(a - b)
    return {"window_a": list(window_a), "window_b": list(window_b),
            "n_in_a": len(a), "n_in_b": len(b),
            "n_added": len(added), "n_removed": len(removed),
            "added_sample": added[:50], "removed_sample": removed[:50],
            "jaccard": round(len(a & b) / len(a | b), 4) if (a | b) else 1.0}


def co_active(world: TemporalWorld, nodes: Sequence[int], *,
              window: tuple[int, int] | None = None, n_bins: int = 40) -> dict[str, Any]:
    """How often the given nodes are active in the same time bin — a coordination probe."""
    lo, hi = window if window is not None else world.horizon
    edges = np.linspace(lo, hi + 1, n_bins + 1).astype(np.int64)
    per_node: dict[int, set[int]] = {n: set() for n in nodes}
    m = world.window_mask(lo, hi)
    for u, v, t in zip(world.src[m].tolist(), world.dst[m].tolist(), world.t[m].tolist()):
        b = int(np.searchsorted(edges, t, side="right") - 1)
        if u in per_node:
            per_node[u].add(b)
        if v in per_node:
            per_node[v].add(b)
    ns = list(nodes)
    pairs = []
    for i in range(len(ns)):
        for j in range(i + 1, len(ns)):
            a, b2 = per_node[ns[i]], per_node[ns[j]]
            if a or b2:
                pairs.append({"pair": [ns[i], ns[j]],
                              "jaccard": round(len(a & b2) / len(a | b2), 4) if (a | b2) else 0.0})
    pairs.sort(key=lambda d: (-d["jaccard"], d["pair"]))     # ties break by node id
    return {"n_nodes": len(ns), "top_pairs": pairs[:30]}


def compute(expression: str, values: dict[str, float] | None = None) -> dict[str, Any]:
    """Tiny arithmetic helper so the agent never has to do mental math on tool output."""
    allowed = {k: v for k, v in (values or {}).items()}
    allowed.update({"abs": abs, "min": min, "max": max, "round": round, "len": len,
                    "sum": sum, "sqrt": math.sqrt, "log": math.log})
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307 - sandboxed names
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"expression": expression, "result": result}
