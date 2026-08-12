"""Background temporal graph generation.

The background must be *hostile to shortcuts*: heavy-tailed node activity (so popularity is a
tempting explanation) and bursty timing (so recency is a tempting explanation). Mechanisms are
planted into this substrate; confounders are layered on top identically in both worlds.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from worlds.base import TAG_BACKGROUND, TemporalWorld


def make_background(n_nodes: int, events_per_node: int, horizon: tuple[int, int], *,
                    rng: np.random.Generator, rate_alpha: float = 1.6,
                    burstiness: float = 0.6, n_burst_centres: int = 12,
                    directed: bool = True) -> TemporalWorld:
    """Heavy-tailed, bursty background activity.

    Node activity rates follow a Pareto law, so a handful of hubs dominate. Event times are a
    mixture of a uniform component and Gaussian bursts, so activity volume fluctuates
    independently of any planted mechanism.
    """
    lo, hi = horizon
    n_events = int(n_nodes * events_per_node)

    rates = rng.pareto(rate_alpha, size=n_nodes) + 1.0
    probs = rates / rates.sum()

    src = rng.choice(n_nodes, size=n_events, p=probs)
    dst = rng.choice(n_nodes, size=n_events, p=probs)
    self_loops = src == dst
    dst[self_loops] = (dst[self_loops] + 1 + rng.integers(0, n_nodes - 1,
                                                          size=int(self_loops.sum()))) % n_nodes

    n_bursty = int(round(burstiness * n_events))
    centres = rng.uniform(lo, hi, size=n_burst_centres)
    width = (hi - lo) / (n_burst_centres * 6.0)
    which = rng.integers(0, n_burst_centres, size=n_bursty)
    t_bursty = rng.normal(centres[which], width)
    t_uniform = rng.uniform(lo, hi, size=n_events - n_bursty)
    t = np.concatenate([t_bursty, t_uniform])
    rng.shuffle(t)
    t = np.clip(np.round(t), lo, hi).astype(np.int64)

    events = [(int(u), int(v), int(tt), TAG_BACKGROUND) for u, v, tt in zip(src, dst, t)]
    return TemporalWorld.from_events(n_nodes, events, horizon, directed=directed)


def low_traffic_nodes(world: TemporalWorld, k: int, *, rng: np.random.Generator,
                      percentile: tuple[float, float] = (20.0, 70.0)) -> list[int]:
    """Sample ``k`` distinct nodes of middling activity.

    Planting on hubs would let a popularity heuristic succeed by accident; planting on isolates
    would make the mechanism trivially visible. Mid-activity nodes avoid both.
    """
    deg = world.degree_vector()
    lo_d, hi_d = np.percentile(deg, percentile)
    pool = np.flatnonzero((deg >= lo_d) & (deg <= hi_d))
    if pool.size < k:
        pool = np.argsort(deg)[: max(k * 10, 50)]
    return [int(x) for x in rng.choice(pool, size=k, replace=False)]
