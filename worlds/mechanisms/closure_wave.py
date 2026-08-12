"""M2 — temporal triadic-closure wave (plan §7-M2).

Both worlds contain the *same* triangles on the *same* edges with the *same* timestamps. In `G+`
each triangle's third edge arrives promptly after its wedge completes, so closure is a temporal
process. In `G-` the identical timestamps are re-assigned within each triangle so the third edge
lands at an arbitrary moment — usually before the wedge is even complete.

A static triangle count is therefore identical in both worlds; only the closing lag differs.
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1].parent))

from worlds.background import low_traffic_nodes, make_background
from worlds.base import TAG_MECHANISM, TemporalWorld
from tools.temporal_surface import temporal_closure

MECHANISM = "closure_wave"


def closure_gold(world: TemporalWorld, *, nodes: list[int], window: tuple[int, int],
                 delta: int, threshold: float) -> tuple[bool, float]:
    """(label, statistic) — the fraction of triangles among ``nodes`` that closed within δ."""
    c = temporal_closure(world, nodes, window=window, delta=delta)
    return (c.fraction >= threshold), float(c.fraction)


def closure_global_check(world: TemporalWorld, *, nodes: list[int], window: tuple[int, int],
                         delta: int, threshold: float) -> str:
    c = temporal_closure(world, nodes, window=window, delta=delta)
    if c.fraction >= threshold:
        return "fail: closure fraction still above threshold in negative world"
    return "pass"


@dataclass
class ClosureSpec:
    nodes: list[int]                     # all nodes participating in planted triangles
    triangles: list[tuple[int, int, int]]
    window: tuple[int, int]
    delta: int
    threshold: float


def generate_closure_pair(cfg: dict[str, Any], seed: int, *, scale: str = "dev"
                          ) -> tuple[TemporalWorld, TemporalWorld, ClosureSpec, int]:
    rng_cfg, m_cfg = cfg["background"], cfg["m2_closure_wave"]
    horizon = tuple(rng_cfg["horizon"])
    n_nodes = rng_cfg["n_nodes_dev"] if scale == "dev" else rng_cfg["n_nodes_test"]
    max_attempts = cfg["generation"]["max_rejection_attempts"]

    n_wedges, delta = m_cfg["n_wedges"], m_cfg["delta"]
    wave_window, threshold = m_cfg["wave_window"], m_cfg["threshold"]
    margin = m_cfg["required_margin"]

    for attempt in range(1, max_attempts + 1):
        sub = np.random.default_rng(seed * 1000 + attempt)
        base = make_background(n_nodes, rng_cfg["events_per_node"], horizon, rng=sub,
                               rate_alpha=rng_cfg["rate_alpha"],
                               burstiness=rng_cfg["burstiness"], directed=rng_cfg["directed"])

        picked = low_traffic_nodes(base, n_wedges * 3, rng=sub)
        triangles = [tuple(picked[3 * i:3 * i + 3]) for i in range(n_wedges)]
        nodes = sorted(picked)

        centre = float(sub.uniform(horizon[0] + 0.25 * (horizon[1] - horizon[0]),
                                   horizon[1] - 0.25 * (horizon[1] - horizon[0])))
        window = (int(centre - wave_window), int(centre + wave_window))

        tri_edges = [[(a, b), (b, c), (a, c)] for (a, b, c) in triangles]

        # positive: each triangle's three timestamps are clustered, so its third edge lands
        # within delta of its second
        pos_times: list[list[int]] = []
        for _ in triangles:
            # t1 must leave room for the whole chain, since t3 can reach t1 + 1.8*delta.
            # Overshooting the window end would push a *different edge* outside the observed
            # window in each world, making the static view differ for a reason that has nothing
            # to do with the mechanism.
            t1 = int(sub.uniform(window[0], window[1] - 2 * delta))
            t2 = int(t1 + sub.uniform(1, delta))
            t3 = int(t2 + sub.uniform(1, delta * 0.8))
            pos_times.append([t1, t2, t3])

        pos_events = [(u, v, t, TAG_MECHANISM)
                      for edges, times in zip(tri_edges, pos_times)
                      for (u, v), t in zip(edges, times)]

        # negative: the same timestamp multiset dealt ACROSS triangles instead of within them.
        #
        # Permuting inside a triangle cannot work: the closure statistic depends on the triangle's
        # three times only through their sorted order, which any within-triangle permutation
        # leaves unchanged. Dealing across triangles keeps the global multiset identical while
        # scattering each triangle's three times far apart, so nothing closes promptly.
        flat = sorted(t for ts in pos_times for t in ts)
        n_tri = len(triangles)
        neg_events = []
        for i, edges in enumerate(tri_edges):
            times_i = [flat[j * n_tri + i] for j in range(3)]
            neg_events += [(u, v, t, TAG_MECHANISM) for (u, v), t in zip(edges, times_i)]

        pos = _plant(base, pos_events)
        neg = _plant(base, neg_events)

        kw = dict(nodes=nodes, window=window, delta=delta, threshold=threshold)
        pos_ok, pos_stat = closure_gold(pos, **kw)
        neg_ok, neg_stat = closure_gold(neg, **kw)

        if (pos_ok and not neg_ok and (pos_stat - threshold) >= margin
                and (threshold - neg_stat) >= margin
                and closure_global_check(neg, **kw) == "pass"):
            spec = ClosureSpec(nodes=nodes, triangles=triangles, window=window, delta=delta,
                               threshold=threshold)
            return pos, neg, spec, attempt

    raise RuntimeError(f"closure seed {seed}: no admissible pair in {max_attempts} attempts")


def _plant(base: TemporalWorld, extra: list[tuple[int, int, int, int]]) -> TemporalWorld:
    w = base.copy()
    src = np.concatenate([w.src, np.array([e[0] for e in extra], dtype=np.int32)])
    dst = np.concatenate([w.dst, np.array([e[1] for e in extra], dtype=np.int32)])
    t = np.concatenate([w.t, np.array([e[2] for e in extra], dtype=np.int64)])
    tag = np.concatenate([w.tag, np.array([e[3] for e in extra], dtype=np.uint8)])
    return TemporalWorld(w.n_nodes, src, dst, t, tag, w.horizon, w.directed, dict(w.meta))
