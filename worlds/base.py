"""Temporal world representation.

A world is a set of timestamped events over a fixed node set. Internally each event carries a
provenance tag (background / mechanism / confounder) that the *served* view must never expose —
serving sorts events canonically by (t, u, v) so that row order cannot reveal which events were
planted (R5).

The counterfactual construction (R1) is a permutation of the timestamp *assignment*: G- reuses the
exact edge multiset and the exact timestamp multiset of G+, re-paired so the target mechanism is
broken. Both worlds therefore agree on every marginal an agent can cheaply observe.
"""

from __future__ import annotations

import csv
import pathlib
import sys
from dataclasses import dataclass, field
from collections import Counter
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import stable_hash

TAG_BACKGROUND = 0
TAG_MECHANISM = 1
TAG_CONFOUNDER = 2
TAG_NAMES = {TAG_BACKGROUND: "background", TAG_MECHANISM: "mechanism",
             TAG_CONFOUNDER: "confounder"}


@dataclass
class TemporalWorld:
    """An event graph. ``src[i] -- dst[i]`` occurred at time ``t[i]``."""

    n_nodes: int
    src: np.ndarray                    # int32
    dst: np.ndarray                    # int32
    t: np.ndarray                      # int64
    tag: np.ndarray                    # uint8, internal only
    horizon: tuple[int, int]
    directed: bool = True
    meta: dict[str, Any] = field(default_factory=dict)   # internal only

    # ---------------------------------------------------------------- backend accounting (C6)
    # Milestone C6 forbids conflating LLM interaction cost with the work the data system does.
    # Interaction cost is already recorded (rounds, calls, tokens, dollars); what was missing is
    # how many events an operator had to examine. Every method below that makes a pass over the
    # event arrays reports the pass, so an interface that looks cheap in tool calls cannot hide a
    # full scan behind each one -- which is exactly A2's shape.
    #
    # Deliberately counted as *events examined*, not bytes or cache lines: it is the unit the
    # operators are written in, it is engine-independent, and it does not pretend to a precision
    # a NumPy mask cannot support.

    def note_scan(self, n: int) -> None:
        self._scanned = getattr(self, "_scanned", 0) + int(n)

    @property
    def scanned(self) -> int:
        """Events examined so far. Deltas around a tool call give that call's backend cost."""
        return getattr(self, "_scanned", 0)

    # ---------------------------------------------------------------- construction
    @staticmethod
    def from_events(n_nodes: int, events: Iterable[tuple[int, int, int, int]],
                    horizon: tuple[int, int], directed: bool = True,
                    meta: dict[str, Any] | None = None) -> "TemporalWorld":
        ev = list(events)
        if not ev:
            raise ValueError("world must contain at least one event")
        arr = np.asarray(ev, dtype=np.int64)
        return TemporalWorld(
            n_nodes=int(n_nodes),
            src=arr[:, 0].astype(np.int32), dst=arr[:, 1].astype(np.int32),
            t=arr[:, 2].astype(np.int64), tag=arr[:, 3].astype(np.uint8),
            horizon=(int(horizon[0]), int(horizon[1])), directed=directed,
            meta=dict(meta or {}),
        )

    def copy(self) -> "TemporalWorld":
        return TemporalWorld(self.n_nodes, self.src.copy(), self.dst.copy(), self.t.copy(),
                             self.tag.copy(), self.horizon, self.directed, dict(self.meta))

    def __len__(self) -> int:
        return int(self.src.shape[0])

    @property
    def n_events(self) -> int:
        return len(self)

    # ---------------------------------------------------------------- invariant surfaces
    def static_edge_multiset(self) -> Counter:
        self.note_scan(len(self))
        """Multiset of (u, v). Undirected worlds normalise orientation."""
        if self.directed:
            pairs = zip(self.src.tolist(), self.dst.tolist())
        else:
            pairs = (tuple(sorted(p)) for p in zip(self.src.tolist(), self.dst.tolist()))
        return Counter(pairs)

    def timestamp_multiset(self) -> Counter:
        self.note_scan(len(self))
        return Counter(self.t.tolist())

    def degree_vector(self) -> np.ndarray:
        self.note_scan(len(self))
        """Event-count degree (a multigraph degree): how often each node participates."""
        deg = np.zeros(self.n_nodes, dtype=np.int64)
        np.add.at(deg, self.src, 1)
        np.add.at(deg, self.dst, 1)
        return deg

    def activity_histogram(self, n_bins: int = 100) -> np.ndarray:
        self.note_scan(len(self))
        lo, hi = self.horizon
        edges = np.linspace(lo, hi + 1, n_bins + 1)
        return np.histogram(self.t, bins=edges)[0]

    def tag_counts(self) -> dict[str, int]:
        c = Counter(self.tag.tolist())
        return {TAG_NAMES[k]: int(v) for k, v in sorted(c.items())}

    # ---------------------------------------------------------------- ordering + hashing
    def canonical_order(self) -> np.ndarray:
        """Sort by (t, src, dst) — independent of provenance tag, so it leaks nothing."""
        return np.lexsort((self.dst, self.src, self.t))

    def sorted_triples(self) -> list[tuple[int, int, int]]:
        self.note_scan(len(self))
        o = self.canonical_order()
        return list(zip(self.src[o].tolist(), self.dst[o].tolist(), self.t[o].tolist()))

    def world_hash(self) -> str:
        """Content hash of the observable world.

        Depends on the (u, v, t) triples, so G+ and G- hash differently, while regenerating from
        the same seed reproduces the hash exactly (T1.1 acceptance test).
        """
        return stable_hash({
            "n": self.n_nodes, "d": self.directed, "h": list(self.horizon),
            "e": self.sorted_triples(),
        })

    # ---------------------------------------------------------------- serving
    def served_events(self) -> np.ndarray:
        """(n_events, 3) int64 array of (u, v, t) in canonical order. No tags, no metadata."""
        o = self.canonical_order()
        return np.column_stack([self.src[o].astype(np.int64), self.dst[o].astype(np.int64),
                                self.t[o]])

    def write_served(self, path: str | pathlib.Path) -> str:
        """Write the agent-visible event table plus a minimal sidecar.

        The sidecar holds only ``n_nodes``, ``horizon`` and ``directed`` — three facts that are
        identical for both members of a matched pair and therefore leak nothing about polarity,
        but which the tool layer needs in order to compute over exactly the same domain as the
        gold oracle.
        """
        import json
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        rows = self.served_events()
        with p.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["u", "v", "t"])
            w.writerows(rows.tolist())
        p.with_suffix(".meta.json").write_text(json.dumps(
            {"n_nodes": int(self.n_nodes), "horizon": list(self.horizon),
             "directed": bool(self.directed)}, sort_keys=True))
        return self.world_hash()

    # ---------------------------------------------------------------- views
    def window_mask(self, lo: int, hi: int) -> np.ndarray:
        self.note_scan(len(self))          # every timestamp is tested
        """Events with lo <= t <= hi."""
        return (self.t >= lo) & (self.t <= hi)

    def events_in(self, lo: int, hi: int) -> np.ndarray:
        m = self.window_mask(lo, hi)
        return np.column_stack([self.src[m].astype(np.int64), self.dst[m].astype(np.int64),
                                self.t[m]])

    def adjacency_at(self, lo: int, hi: int) -> dict[int, set[int]]:
        """Static neighbour sets induced by events in [lo, hi]."""
        adj: dict[int, set[int]] = {}
        m = self.window_mask(lo, hi)
        for u, v in zip(self.src[m].tolist(), self.dst[m].tolist()):
            adj.setdefault(u, set()).add(v)
            if not self.directed:
                adj.setdefault(v, set()).add(u)
        return adj

    def out_events_by_node(self) -> dict[int, list[tuple[int, int]]]:
        self.note_scan(len(self))
        """node -> [(t, neighbour)] sorted by t. Used by the temporal reachability oracle."""
        out: dict[int, list[tuple[int, int]]] = {}
        for u, v, t in zip(self.src.tolist(), self.dst.tolist(), self.t.tolist()):
            out.setdefault(u, []).append((t, v))
            if not self.directed:
                out.setdefault(v, []).append((t, u))
        for lst in out.values():
            lst.sort()
        return out
