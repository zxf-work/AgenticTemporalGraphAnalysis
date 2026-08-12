"""M3 — abrupt neighbourhood regime switch (plan §7-M3).

A focal group interacts with one set of partners, then with another. Both worlds contain the *same*
interactions with the *same* timestamps; in `G+` the old-partner events all fall before a change
point and the new-partner events all fall after it, so the turnover is abrupt. In `G-` the same
timestamps are re-assigned across the same edges so old and new partners are interleaved
throughout, producing gradual churn.

Endpoint structure is identical: the same partners, the same event counts, the same final
neighbourhood. Only *when* each partner was active differs.
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
from tools.temporal_surface import neighborhood_evolution

MECHANISM = "regime_switch"


def regime_gold(world: TemporalWorld, *, group: list[int], window: tuple[int, int],
                bandwidth: int, threshold: float) -> tuple[bool, float]:
    """(label, statistic) — the peak of ``1 - Jaccard(contacts before, contacts after)``."""
    r = neighborhood_evolution(world, group, window=window, bandwidth=bandwidth)
    return (r.peak_change >= threshold), float(r.peak_change)


def regime_global_check(world: TemporalWorld, *, group: list[int], window: tuple[int, int],
                        bandwidth: int, threshold: float) -> str:
    r = neighborhood_evolution(world, group, window=window, bandwidth=bandwidth)
    if r.peak_change >= threshold:
        return "fail: an abrupt turnover is still present in the negative world"
    return "pass"


@dataclass
class RegimeSpec:
    group: list[int]
    old_partners: list[int]
    new_partners: list[int]
    change_point: int
    window: tuple[int, int]
    bandwidth: int
    threshold: float


def generate_regime_pair(cfg: dict[str, Any], seed: int, *, scale: str = "dev"
                         ) -> tuple[TemporalWorld, TemporalWorld, RegimeSpec, int]:
    bg_cfg, m_cfg = cfg["background"], cfg["m3_regime_switch"]
    horizon = tuple(bg_cfg["horizon"])
    n_nodes = bg_cfg["n_nodes_dev"] if scale == "dev" else bg_cfg["n_nodes_test"]
    max_attempts = cfg["generation"]["max_rejection_attempts"]

    gsize, n_old, n_new = m_cfg["group_size"], m_cfg["old_partners"], m_cfg["new_partners"]
    transition, bandwidth = m_cfg["transition_window"], m_cfg["bandwidth"]
    threshold, margin = m_cfg["threshold"], m_cfg["required_margin"]

    for attempt in range(1, max_attempts + 1):
        sub = np.random.default_rng(seed * 1000 + attempt)
        base = make_background(n_nodes, bg_cfg["events_per_node"], horizon, rng=sub,
                               rate_alpha=bg_cfg["rate_alpha"],
                               burstiness=bg_cfg["burstiness"], directed=bg_cfg["directed"])

        picked = low_traffic_nodes(base, gsize + n_old + n_new, rng=sub)
        group = sorted(picked[:gsize])
        old_p = sorted(picked[gsize:gsize + n_old])
        new_p = sorted(picked[gsize + n_old:])

        span = bandwidth * 2
        tc = int(sub.uniform(horizon[0] + span + bandwidth, horizon[1] - span - bandwidth))
        window = (max(horizon[0], tc - span - bandwidth), min(horizon[1], tc + span + bandwidth))

        # edge multiset: every group member contacts every old partner, then every new partner,
        # each pair interacting `repeats` times
        reps = m_cfg.get("repeats", 1)
        old_edges = [(g, p) for g in group for p in old_p for _ in range(reps)]
        new_edges = [(g, p) for g in group for p in new_p for _ in range(reps)]

        # timestamp multiset: half before the change point, half after
        t_before = sub.uniform(window[0], tc - transition / 2, size=len(old_edges))
        t_after = sub.uniform(tc + transition / 2, window[1], size=len(new_edges))
        times = np.concatenate([t_before, t_after]).round().astype(int)

        edges = old_edges + new_edges
        pos_events = [(u, v, int(t), TAG_MECHANISM) for (u, v), t in zip(edges, times)]

        # negative: identical edges, identical timestamps, dealt so that EVERY partner is active
        # throughout the period rather than confined to an era.
        #
        # A uniformly random reshuffle is not enough: each observation window would still sample a
        # different random subset of partners, leaving substantial apparent turnover. Dealing the
        # sorted timestamps round-robin over partners gives each partner an evenly spread set of
        # times, so the contacts before and after any probe point are nearly the same set and the
        # turnover statistic collapses.
        by_partner: dict[int, list[tuple[int, int]]] = {}
        for (g, p) in edges:
            by_partner.setdefault(p, []).append((g, p))
        partners = sorted(by_partner)
        interleaved: list[tuple[int, int]] = []
        for i in range(max(len(v) for v in by_partner.values())):
            for p in partners:
                if i < len(by_partner[p]):
                    interleaved.append(by_partner[p][i])
        neg_events = [(u, v, int(t), TAG_MECHANISM)
                      for (u, v), t in zip(interleaved, sorted(times))]

        pos = _plant(base, pos_events)
        neg = _plant(base, neg_events)

        kw = dict(group=group, window=window, bandwidth=bandwidth, threshold=threshold)
        pos_ok, pos_stat = regime_gold(pos, **kw)
        neg_ok, neg_stat = regime_gold(neg, **kw)

        if (pos_ok and not neg_ok and (pos_stat - threshold) >= margin
                and (threshold - neg_stat) >= margin
                and regime_global_check(neg, **kw) == "pass"):
            spec = RegimeSpec(group=group, old_partners=old_p, new_partners=new_p,
                              change_point=tc, window=window, bandwidth=bandwidth,
                              threshold=threshold)
            return pos, neg, spec, attempt

    raise RuntimeError(f"regime seed {seed}: no admissible pair in {max_attempts} attempts")


def _plant(base: TemporalWorld, extra: list[tuple[int, int, int, int]]) -> TemporalWorld:
    w = base.copy()
    src = np.concatenate([w.src, np.array([e[0] for e in extra], dtype=np.int32)])
    dst = np.concatenate([w.dst, np.array([e[1] for e in extra], dtype=np.int32)])
    t = np.concatenate([w.t, np.array([e[2] for e in extra], dtype=np.int64)])
    tag = np.concatenate([w.tag, np.array([e[3] for e in extra], dtype=np.uint8)])
    return TemporalWorld(w.n_nodes, src, dst, t, tag, w.horizon, w.directed, dict(w.meta))
