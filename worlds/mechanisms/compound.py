"""M4 — compound temporal investigation (milestone C5).

Not a fourth simple mechanism; the stop rule after M3 forbids that. This is a **two-stage
investigation with a dependency between the stages**:

    ClosureWave(C, W, delta)  completes at  t_c
    Cascade(s -> t, max_wait) lies entirely within  [t_c, t_c + admissible]

The whole point is the *binding of the second stage's window to the first stage's result*. So the
counterfactual is built on that binding and nothing else:

    G+  closure wave present, cascade present and time-respecting, INSIDE  [t_c, t_c + admissible]
    G-  closure wave present, cascade present and time-respecting, OUTSIDE that interval

Both worlds contain **both stages, individually satisfied**. An agent that verifies "is there a
closure wave?" and "is there a cascade?" and conjoins the answers gets the negative world wrong
every time. Only an agent that computes t_c and then constrains the cascade search to the interval
that follows it can separate the pair. That is precisely the compositional competence C5 exists to
measure, and it is invisible to any single-predicate benchmark.

Matched-pair discipline is unchanged and, if anything, tighter than the single-stage mechanisms:
identical node set, identical static edge multiset, identical event count, and an identical global
timestamp multiset. The cascade is *moved in time* by swapping its timestamp assignment with that
of an equal number of background events, so the multiset is preserved exactly rather than
approximately. Only the assignment of times to edges differs between G+ and G-.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from tools.temporal_surface import temporal_closure, temporal_reachability
from worlds.background import low_traffic_nodes, make_background
from worlds.base import TAG_MECHANISM, TemporalWorld


@dataclass
class CompoundSpec:
    group: list[int]                       # C, the group whose closure wave is stage 1
    triangles: list[tuple[int, int, int]]
    source: int                            # s, stage 2
    target: int                            # t
    path_nodes: list[int] = field(default_factory=list)
    window: tuple[int, int] = (0, 0)       # the observation window served to the agent
    delta: int = 0                         # stage-1 closure bound
    max_wait: int = 0                      # stage-2 waiting bound
    admissible: int = 0                    # stage-2 must start within this long after t_c
    threshold: float = 0.5
    closure_threshold: float = 0.45


# ======================================================================================
# gold
# ======================================================================================

def closure_completion_time(world: TemporalWorld, *, group: list[int], window: tuple[int, int],
                            delta: int) -> int | None:
    """t_c — when the closure wave finishes: the last closing edge among δ-closed triangles.

    Defined on the observable world alone, so the agent can in principle compute exactly what the
    oracle computes. Returns None when no triangle closes within δ, in which case stage 1 did not
    happen and the compound question is answered "absent" without reaching stage 2.
    """
    c = temporal_closure(world, group, window=window, delta=delta)
    if not c.closures:
        return None
    return int(max(t_close for *_rest, t_close in c.closures))


def compound_gold(world: TemporalWorld, *, group: list[int], window: tuple[int, int], delta: int,
                  closure_threshold: float, source: int, target: int, max_wait: int,
                  admissible: int) -> tuple[bool, float]:
    """(label, statistic). Both stages must hold AND the second must follow the first in time."""
    c = temporal_closure(world, group, window=window, delta=delta)
    if c.fraction < closure_threshold:
        return False, 0.0                                   # stage 1 absent
    t_c = closure_completion_time(world, group=group, window=window, delta=delta)
    if t_c is None:
        return False, 0.0
    r = temporal_reachability(world, source, target, window=(t_c, min(window[1],
                                                                     t_c + admissible)),
                              max_wait=max_wait)
    # 1.0 both stages and the dependency; 0.0 otherwise. The mechanism is a conjunction, so a
    # graded statistic would invent a precision the question does not have -- as with M1, which is
    # also a predicate rather than a fraction.
    return (True, 1.0) if r.reached else (False, 0.0)


def compound_global_check(world: TemporalWorld, *, group: list[int], window: tuple[int, int],
                          delta: int, closure_threshold: float, source: int, target: int,
                          max_wait: int, admissible: int) -> str:
    """The negative world must not satisfy the compound predicate by some other route."""
    ok, _ = compound_gold(world, group=group, window=window, delta=delta,
                          closure_threshold=closure_threshold, source=source, target=target,
                          max_wait=max_wait, admissible=admissible)
    if ok:
        return "fail: compound predicate still satisfied in the negative world"
    return "pass"


# ======================================================================================
# generation
# ======================================================================================

def _plant(base: TemporalWorld, extra: list[tuple[int, int, int, int]]) -> TemporalWorld:
    w = base.copy()
    src = np.concatenate([w.src, np.array([e[0] for e in extra], dtype=np.int32)])
    dst = np.concatenate([w.dst, np.array([e[1] for e in extra], dtype=np.int32)])
    t = np.concatenate([w.t, np.array([e[2] for e in extra], dtype=np.int64)])
    tag = np.concatenate([w.tag, np.array([e[3] for e in extra], dtype=np.uint8)])
    return TemporalWorld(w.n_nodes, src, dst, t, tag, w.horizon, w.directed, dict(w.meta))


def _closure_events(triangles: list[tuple[int, int, int]], start: int, delta: int,
                    rng: np.random.Generator) -> tuple[list[tuple[int, int, int, int]], int]:
    """Plant a closure wave: each triangle's third edge lands within δ of its wedge completing."""
    ev: list[tuple[int, int, int, int]] = []
    last_close = start
    for i, (a, b, c) in enumerate(triangles):
        base_t = int(start + i * max(1, delta // 2))
        t1 = base_t
        t2 = base_t + int(rng.integers(1, max(2, delta // 4)))
        t3 = t2 + int(rng.integers(1, max(2, delta)))        # closes within delta of the wedge
        ev += [(a, b, t1, TAG_MECHANISM), (b, c, t2, TAG_MECHANISM), (a, c, t3, TAG_MECHANISM)]
        last_close = max(last_close, t3)
    return ev, last_close


def generate_compound_pair(cfg: dict[str, Any], seed: int, *, scale: str = "dev"
                           ) -> tuple[TemporalWorld, TemporalWorld, CompoundSpec, int]:
    rng = np.random.default_rng(seed)
    bg_cfg, m_cfg = cfg["background"], cfg["m4_compound"]
    horizon = tuple(bg_cfg["horizon"])
    n_nodes = bg_cfg["n_nodes_dev"] if scale == "dev" else bg_cfg["n_nodes_test"]
    max_attempts = cfg["generation"]["max_rejection_attempts"]

    n_tri = m_cfg["n_triangles"]
    delta = m_cfg["delta"]
    hops, max_wait = m_cfg["hops"], m_cfg["max_wait"]
    admissible = m_cfg["admissible"]
    closure_threshold = m_cfg["closure_threshold"]
    late_offset = m_cfg["late_offset"]           # how far past the interval G- puts the cascade

    for attempt in range(1, max_attempts + 1):
        sub = np.random.default_rng(seed * 7919 + attempt)
        base = make_background(n_nodes, bg_cfg["events_per_node"], horizon, rng=sub,
                               rate_alpha=bg_cfg["rate_alpha"], burstiness=bg_cfg["burstiness"],
                               directed=bg_cfg["directed"])

        picked = low_traffic_nodes(base, 3 * n_tri + hops + 1, rng=sub)
        tri_nodes = picked[:3 * n_tri]
        triangles = [(tri_nodes[3 * i], tri_nodes[3 * i + 1], tri_nodes[3 * i + 2])
                     for i in range(n_tri)]
        path_nodes = picked[3 * n_tri:3 * n_tri + hops + 1]

        span = horizon[1] - horizon[0]
        wave_start = int(horizon[0] + 0.20 * span)
        closure_ev, t_c = _closure_events(triangles, wave_start, delta, sub)

        # stage-2 timings: identical inter-hop gaps in both worlds, so the cascade is equally
        # time-respecting either way and only its POSITION relative to t_c differs.
        gaps = [int(sub.integers(max_wait // 4, max_wait // 2)) for _ in range(hops - 1)]
        in_start = int(t_c + admissible * 0.25)
        out_start = int(t_c + admissible + late_offset)
        t_in, t_out = [in_start], [out_start]
        for g in gaps:
            t_in.append(t_in[-1] + g)
            t_out.append(t_out[-1] + g)
        if t_out[-1] > horizon[1] - 0.05 * span:
            continue                                   # the late cascade would fall outside view

        cascade_edges = [(int(path_nodes[i]), int(path_nodes[i + 1])) for i in range(hops)]

        # Preserve the global timestamp multiset exactly: the two time-sets are swapped between
        # the cascade edges and an equal number of background events. Whatever G+ gives the
        # cascade, G- gives those background events, and vice versa.
        n_swap = len(cascade_edges)
        idx = sub.choice(len(base), size=n_swap, replace=False)
        donor_times = base.t[idx].tolist()

        def build(cascade_times: list[int], donor_gets: list[int]) -> TemporalWorld:
            b = base.copy()
            b.t[idx] = np.array(donor_gets, dtype=np.int64)
            ev = list(closure_ev)
            ev += [(u, v, int(tt), TAG_MECHANISM)
                   for (u, v), tt in zip(cascade_edges, cascade_times)]
            return _plant(b, ev)

        # G+: cascade takes the in-window times, donors take the out times (and vice versa), so
        # both worlds carry the identical multiset {t_in} u {t_out} u {rest}.
        pos = build(t_in, t_out)
        neg = build(t_out, t_in)
        # the donor events keep their original times in neither world, but they keep the SAME
        # pair of assignments across the two, which is what the invariant checks
        del donor_times

        window = (max(horizon[0], wave_start - 2 * delta),
                  min(horizon[1], max(t_in[-1], t_out[-1]) + 2 * max_wait))
        spec = CompoundSpec(group=sorted(set(tri_nodes)), triangles=triangles,
                            source=path_nodes[0], target=path_nodes[-1], path_nodes=path_nodes,
                            window=window, delta=delta, max_wait=max_wait,
                            admissible=admissible, threshold=m_cfg["threshold"],
                            closure_threshold=closure_threshold)

        kw = dict(group=spec.group, window=window, delta=delta,
                  closure_threshold=closure_threshold, source=spec.source, target=spec.target,
                  max_wait=max_wait, admissible=admissible)
        pos_ok, pos_stat = compound_gold(pos, **kw)
        neg_ok, neg_stat = compound_gold(neg, **kw)

        # Both worlds must contain BOTH stages individually -- that is what makes the pair a test
        # of composition rather than of either stage. A negative that simply lacks a cascade would
        # be solvable by the single-predicate strategy the workload is designed to defeat.
        neg_closure = temporal_closure(neg, spec.group, window=window, delta=delta)
        neg_cascade_anywhere = temporal_reachability(neg, spec.source, spec.target, window=window,
                                                     max_wait=max_wait).reached
        if (pos_ok and not neg_ok
                and neg_closure.fraction >= closure_threshold
                and neg_cascade_anywhere
                and compound_global_check(neg, **kw) == "pass"):
            return pos, neg, spec, attempt

    raise RuntimeError(f"compound seed {seed}: no admissible pair in {max_attempts} attempts")
