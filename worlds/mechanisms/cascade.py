"""M1 — time-respecting cascade (plan §7-M1).

Both worlds contain the *same* path edges with the *same* timestamps. In `G+` the timestamps are
assigned along the path in increasing order with every gap inside the maximum waiting time, so a
time-respecting cascade exists. In `G-` the identical timestamps are re-assigned among the same
edges so that ordering is violated and no time-respecting route survives.

A static analysis sees one identical path in both worlds. Only an ordering-aware analysis
separates them.
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1].parent))

from worlds.background import low_traffic_nodes, make_background
from worlds.base import TAG_CONFOUNDER, TAG_MECHANISM, TemporalWorld
from tools.temporal_surface import temporal_reachability

MECHANISM = "cascade"


# ======================================================================================
# gold predicate
# ======================================================================================

def cascade_gold(world: TemporalWorld, *, source: int, target: int,
                 window: tuple[int, int], max_wait: int) -> tuple[bool, float]:
    """(label, statistic in [0, 1]) for the cascade predicate.

    The label is plain time-respecting reachability. The statistic measures how *robust* that
    verdict is, so that neither world of a pair sits on the decision boundary (R2):

    * reachable  → ``0.5 + 0.5 * headroom``, where headroom is the unused fraction of the waiting
      budget on the tightest hop of the earliest-arrival path;
    * unreachable → ``0.5 - 0.5 * violation``, where violation grows with how far the waiting
      budget would have to be relaxed before the target became reachable (1.0 if no relaxation
      ever suffices).
    """
    res = temporal_reachability(world, source, target, window=window, max_wait=max_wait)
    if res.reached:
        headroom = 1.0 - (res.max_wait_used / max_wait if max_wait else 0.0)
        return True, 0.5 + 0.5 * float(np.clip(headroom, 0.0, 1.0))

    violation = _relaxation_needed(world, source, target, window, max_wait)
    return False, 0.5 - 0.5 * float(np.clip(violation, 0.0, 1.0))


def _relaxation_needed(world: TemporalWorld, source: int, target: int,
                       window: tuple[int, int], max_wait: int, n_steps: int = 6) -> float:
    """How much the waiting budget must grow before the target becomes reachable.

    Returns ``(Δ' - Δ) / Δ`` capped at 1.0, or 1.0 when even an unbounded wait fails (which is the
    common case for a properly broken cascade: the path is out of order, so no amount of patience
    helps).
    """
    if not temporal_reachability(world, source, target, window=window, max_wait=None).reached:
        return 1.0
    lo_mult, hi_mult = 1.0, 2.0
    for _ in range(n_steps):
        mid = (lo_mult + hi_mult) / 2
        if temporal_reachability(world, source, target, window=window,
                                 max_wait=int(max_wait * mid)).reached:
            hi_mult = mid
        else:
            lo_mult = mid
    return hi_mult - 1.0


def cascade_global_check(world: TemporalWorld, *, source: int, target: int,
                         window: tuple[int, int], max_wait: int) -> str:
    """Reject a negative world in which the target is still reachable by *any* route (R2).

    Without this, a background edge sequence could quietly re-create the mechanism the permutation
    was supposed to destroy, and the pair's gold labels would be wrong.
    """
    res = temporal_reachability(world, source, target, window=window, max_wait=max_wait)
    if res.reached:
        return "fail: target reachable in negative world via alternative route"
    return "pass"


# ======================================================================================
# generation
# ======================================================================================

@dataclass
class CascadeSpec:
    source: int
    target: int
    path_nodes: list[int]
    times: list[int]
    window: tuple[int, int]
    max_wait: int


def generate_cascade_pair(cfg: dict[str, Any], seed: int, *, scale: str = "dev"
                          ) -> tuple[TemporalWorld, TemporalWorld, CascadeSpec, int]:
    """Build a matched (G+, G-) cascade pair. Returns (positive, negative, spec, attempts)."""
    rng = np.random.default_rng(seed)
    bg_cfg, m_cfg = cfg["background"], cfg["m1_cascade"]
    horizon = tuple(bg_cfg["horizon"])
    n_nodes = bg_cfg["n_nodes_dev"] if scale == "dev" else bg_cfg["n_nodes_test"]
    max_attempts = cfg["generation"]["max_rejection_attempts"]

    hops, max_wait = m_cfg["hops"], m_cfg["max_wait"]
    slack_lo, slack_hi = m_cfg["slack_fraction"]
    required_margin = m_cfg["required_margin"]
    threshold = m_cfg["threshold"]

    n_decoys = m_cfg.get("n_decoy_paths", 0)

    # ---- temporal resolution R (T17) --------------------------------------------------
    # R is the number of distinguishable temporal positions an agent must search across the
    # observation window: R = span / unit, with unit = max_wait/8, the granularity at which the
    # waiting bound can still be decided.
    #
    # R is implemented by widening the *window*, not by quantising timestamps. Quantising would
    # collapse the cascade's inter-hop gaps into ties at low R, destroying the mechanism instead
    # of merely making it harder to find, and the axis would degenerate rather than scale.
    #
    # The horizon and the background event count scale with the span so that background *density*
    # stays constant; otherwise a larger R would silently also mean a sparser graph, and the axis
    # would confound search cost with density.
    resolution = m_cfg.get("temporal_resolution")
    unit = max(1, max_wait // 8)
    events_per_node = bg_cfg["events_per_node"]
    if resolution:
        span_target = int(resolution) * unit
        base_span = horizon[1] - horizon[0]
        need = int(span_target * 2.5)
        if need > base_span:
            scale_factor = need / base_span
            horizon = (horizon[0], horizon[0] + need)
            events_per_node = max(1, int(round(events_per_node * scale_factor)))
    else:
        span_target = None

    for attempt in range(1, max_attempts + 1):
        sub = np.random.default_rng(seed * 1000 + attempt)
        base = make_background(n_nodes, events_per_node, horizon, rng=sub,
                               rate_alpha=bg_cfg["rate_alpha"],
                               burstiness=bg_cfg["burstiness"],
                               directed=bg_cfg["directed"])

        # one true path plus n_decoys structurally identical alternatives between the same
        # endpoints; interior nodes are disjoint so each route is distinguishable
        needed = (hops + 1) + n_decoys * (hops - 1)
        picked = low_traffic_nodes(base, needed, rng=sub)
        path_nodes = picked[:hops + 1]
        gaps = sub.uniform(slack_lo, slack_hi, size=hops) * max_wait
        span = float(gaps.sum())
        start = float(sub.uniform(horizon[0] + 0.15 * (horizon[1] - horizon[0]),
                                  horizon[1] - span - 0.15 * (horizon[1] - horizon[0])))
        times = [int(round(start + gaps[:i].sum())) for i in range(hops)]
        # The observed window is padded so the agent must locate the cascade rather than read its
        # bounds off the window. When a resolution is requested, the padding is set so the window
        # spans exactly R units of localisation granularity.
        duration = times[-1] - times[0]
        if span_target is not None:
            if span_target <= duration:
                raise RuntimeError(
                    f"temporal_resolution={resolution} gives a span of {span_target} but the "
                    f"cascade itself lasts {duration}; raise R or shorten the cascade")
            pad = (span_target - duration) // 2
        else:
            pad = max_wait * 3
        window = (max(horizon[0], times[0] - pad), min(horizon[1], times[-1] + pad))

        spec = CascadeSpec(source=path_nodes[0], target=path_nodes[-1], path_nodes=path_nodes,
                           times=times, window=window, max_wait=max_wait)

        decoys = _decoy_events(picked[hops + 1:], spec.source, spec.target, hops, n_decoys,
                               window, sub)

        pos = _plant(base, path_nodes, times, decoys)
        neg_times = _break_order(times, sub, m_cfg["min_order_violations"])
        neg = _plant(base, path_nodes, neg_times, decoys)

        gold_kw = dict(source=spec.source, target=spec.target, window=window, max_wait=max_wait)
        pos_ok, pos_stat = cascade_gold(pos, **gold_kw)
        neg_ok, neg_stat = cascade_gold(neg, **gold_kw)

        if (pos_ok and not neg_ok
                and (pos_stat - threshold) >= required_margin
                and (threshold - neg_stat) >= required_margin
                and cascade_global_check(neg, **gold_kw) == "pass"):
            return pos, neg, spec, attempt

    raise RuntimeError(f"cascade seed {seed}: no admissible pair in {max_attempts} attempts")


def _decoy_events(pool: list[int], source: int, target: int, hops: int, n_decoys: int,
                  window: tuple[int, int], rng: np.random.Generator
                  ) -> list[tuple[int, int, int, int]]:
    """Alternative source->target routes that are never time-respecting.

    Each decoy's timestamps strictly *decrease* along the route, so no amount of waiting-budget
    relaxation makes it traversable and it can never manufacture the mechanism in either world.
    Decoys are tagged as confounder events and are byte-identical in the two worlds, so the
    existing ``same_confounder_events`` invariant checks that automatically.
    """
    out: list[tuple[int, int, int, int]] = []
    per = hops - 1
    for d in range(n_decoys):
        interior = pool[d * per:(d + 1) * per]
        if len(interior) < per:
            break
        route = [source, *interior, target]
        ts = sorted(int(t) for t in rng.uniform(window[0], window[1], size=len(route) - 1))
        ts.reverse()                                   # strictly decreasing => never traversable
        out += [(int(route[i]), int(route[i + 1]), ts[i], TAG_CONFOUNDER)
                for i in range(len(route) - 1)]
    return out


def _plant(base: TemporalWorld, path_nodes: list[int], times: list[int],
           decoys: list[tuple[int, int, int, int]] | None = None) -> TemporalWorld:
    """Add the path edges with the given timestamp assignment, plus any decoy routes."""
    w = base.copy()
    extra = [(int(path_nodes[i]), int(path_nodes[i + 1]), int(times[i]), TAG_MECHANISM)
             for i in range(len(times))]
    extra += list(decoys or [])
    src = np.concatenate([w.src, np.array([e[0] for e in extra], dtype=np.int32)])
    dst = np.concatenate([w.dst, np.array([e[1] for e in extra], dtype=np.int32)])
    t = np.concatenate([w.t, np.array([e[2] for e in extra], dtype=np.int64)])
    tag = np.concatenate([w.tag, np.array([e[3] for e in extra], dtype=np.uint8)])
    return TemporalWorld(w.n_nodes, src, dst, t, tag, w.horizon, w.directed, dict(w.meta))


def _break_order(times: list[int], rng: np.random.Generator, min_violations: int) -> list[int]:
    """Re-assign the *same* timestamps among the *same* path edges, out of order.

    This is the formal content of the counterfactual (R1): the timestamp multiset is untouched, so
    the global activity histogram, the static edge multiset and every node's event count are all
    preserved exactly. Only which edge carries which instant changes.
    """
    for _ in range(50):
        perm = rng.permutation(len(times))
        cand = [times[i] for i in perm]
        violations = sum(1 for i in range(len(cand) - 1) if cand[i + 1] <= cand[i])
        if violations >= min_violations:
            return cand
    return list(reversed(times))       # guaranteed maximal violation
