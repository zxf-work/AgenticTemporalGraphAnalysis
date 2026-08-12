"""Confounder overlays — nuisance activity designed to bait a shortcut heuristic.

The matched-pair construction already neutralises the obvious shortcuts *by construction*: both
worlds carry the same node set, the same static edge multiset, the same event count and the same
global timestamp multiset, so any strategy keyed on activity volume, node popularity or static
connectivity scores exactly 0.00. Confounders are therefore not the primary control; they are a
stress test of that claim, and they answer the reviewer who asks what happens when the bait is
made loud rather than merely present.

The overlay is applied **identically to both worlds** (plan §8). That is the whole design: extra
events are generated once and added to G+ and G- with the same endpoints and the same timestamps,
so they cannot shift the pair's decision and cannot break any invariant. They only make the world
noisier in a way that a volume- or recency-driven agent should find tempting. Overlay events are
tagged ``TAG_CONFOUNDER``, which the pair invariant checker verifies is byte-identical across the
two worlds.

    c1_global_burst   a broad spike in overall activity     baits "lots happened => mechanism"
    c2_hub_burst      a spike concentrated on a few hubs    baits "the hub is busy => mechanism"
    c3_repeat_flood   heavy repetition on a few edges       baits "repetition => structure"
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from worlds.base import TAG_CONFOUNDER, TemporalWorld


def _add(world: TemporalWorld, ev: list[tuple[int, int, int]]) -> TemporalWorld:
    """Append tagged confounder events to a copy of ``world``."""
    if not ev:
        return world.copy()
    w = world.copy()
    src = np.concatenate([w.src, np.array([e[0] for e in ev], dtype=np.int32)])
    dst = np.concatenate([w.dst, np.array([e[1] for e in ev], dtype=np.int32)])
    t = np.concatenate([w.t, np.array([e[2] for e in ev], dtype=np.int64)])
    tag = np.concatenate([w.tag, np.full(len(ev), TAG_CONFOUNDER, dtype=np.uint8)])
    return TemporalWorld(w.n_nodes, src, dst, t, tag, w.horizon, w.directed, dict(w.meta))


def _burst_centre(world: TemporalWorld, rng: np.random.Generator, width: int) -> int:
    """Place the burst inside the observable horizon with room for its width."""
    lo, hi = int(world.horizon[0]), int(world.horizon[1])
    return int(rng.integers(lo + width, max(lo + width + 1, hi - width)))


def _global_burst(world: TemporalWorld, cfg: dict[str, Any], rng: np.random.Generator
                  ) -> list[tuple[int, int, int]]:
    """A broad spike in activity across the whole node set.

    Volume is expressed as a multiplier on the local background rate: `multiplier` times the mean
    number of events that would fall in a window of `width` anyway. That keeps the overlay's size
    proportional to the world rather than a magic constant, so the same setting means the same
    thing at dev and test scale.
    """
    width, mult = int(cfg["width"]), float(cfg["multiplier"])
    span = max(1, int(world.horizon[1]) - int(world.horizon[0]))
    baseline = len(world) * width / span
    n = int(round(baseline * (mult - 1.0)))
    if n <= 0:
        return []
    centre = _burst_centre(world, rng, width)
    times = rng.integers(centre - width // 2, centre + width // 2, size=n)
    u = rng.integers(0, world.n_nodes, size=n)
    v = rng.integers(0, world.n_nodes, size=n)
    return [(int(a), int(b), int(t)) for a, b, t in zip(u, v, times) if a != b]


def _hub_burst(world: TemporalWorld, cfg: dict[str, Any], rng: np.random.Generator
               ) -> list[tuple[int, int, int]]:
    """A spike concentrated on the busiest nodes, baiting a popularity heuristic."""
    width, mult, n_hubs = int(cfg["width"]), float(cfg["multiplier"]), int(cfg["n_hubs"])
    deg = world.degree_vector()
    hubs = np.argsort(deg)[-n_hubs:]
    span = max(1, int(world.horizon[1]) - int(world.horizon[0]))
    baseline = len(world) * width / span
    n = int(round(baseline * (mult - 1.0)))
    if n <= 0:
        return []
    centre = _burst_centre(world, rng, width)
    times = rng.integers(centre - width // 2, centre + width // 2, size=n)
    u = rng.choice(hubs, size=n)
    v = rng.integers(0, world.n_nodes, size=n)
    return [(int(a), int(b), int(t)) for a, b, t in zip(u, v, times) if a != b]


def _repeat_flood(world: TemporalWorld, cfg: dict[str, Any], rng: np.random.Generator
                  ) -> list[tuple[int, int, int]]:
    """Heavy repetition on a handful of edges, baiting "repetition implies structure"."""
    width = int(cfg["width"])
    n_edges, repeats = int(cfg["n_edges"]), int(cfg["repeats"])
    centre = _burst_centre(world, rng, width)
    out: list[tuple[int, int, int]] = []
    for _ in range(n_edges):
        a, b = int(rng.integers(0, world.n_nodes)), int(rng.integers(0, world.n_nodes))
        if a == b:
            continue
        for t in rng.integers(centre - width // 2, centre + width // 2, size=repeats):
            out.append((a, b, int(t)))
    return out


KINDS = {"global_burst": _global_burst, "hub_burst": _hub_burst, "repeat_flood": _repeat_flood}


def apply_confounder(name: str, pos: TemporalWorld, neg: TemporalWorld, cfg: dict[str, Any],
                     seed: int) -> tuple[TemporalWorld, TemporalWorld]:
    """Overlay the named confounder on BOTH worlds identically.

    ``name`` selects an entry of ``cfg["confounders"]`` and a severity, written ``<key>_<level>``
    (for example ``c1_global_burst_high``); a bare key uses ``low``. The overlay is generated once
    from a seed derived from the pair seed and added unchanged to each world, so the two remain
    matched on every declared nuisance statistic including the confounder events themselves.
    """
    if name == "none":
        return pos, neg

    key, _, level = name.rpartition("_")
    if key not in cfg["confounders"]:
        key, level = name, "low"
    spec = cfg["confounders"].get(key)
    if spec is None:
        raise KeyError(f"unknown confounder {name!r}; available: {sorted(cfg['confounders'])}")
    params = spec.get(level or "low")
    if params is None:
        raise KeyError(f"confounder {key!r} has no severity {level!r}")

    fn = KINDS.get(spec["kind"])
    if fn is None:
        raise KeyError(f"confounder kind {spec['kind']!r} is not implemented")

    # One RNG, one event list, added to both worlds -- identity across the pair is the invariant
    # that makes this a nuisance rather than a signal.
    rng = np.random.default_rng(seed * 104729 + 17)
    ev = fn(pos, params, rng)
    return _add(pos, ev), _add(neg, ev)
