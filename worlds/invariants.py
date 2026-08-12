"""Pair invariant record and checker (taskboard Part B3, ticket T1.2).

A pair is admissible only if `G-` is a *timestamp-assignment permutation* of `G+` (R1) and the gold
predicate separates the two worlds by a margin (R2). The checker verifies rather than enforces:
the generators construct pairs by permutation, so a failure here signals a generator bug.

Checked properties
------------------
1.  identical node count and directedness
2.  identical static edge multiset            — an agent cannot separate worlds on structure
3.  identical event count
4.  identical global timestamp multiset       — nor on when activity happened overall
5.  identical per-node event degree           — nor on who was active how often
6.  identical confounder event multiset       — the same distractors in both worlds
7.  gold predicate differs (+ True, - False)
8.  both margins >= the mechanism's required margin
9.  the negative world is globally clean: the mechanism must not have reappeared elsewhere
"""

from __future__ import annotations

import pathlib
import sys
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import SchemaError, check_type, require_keys
from worlds.base import TAG_CONFOUNDER, TemporalWorld


@dataclass
class InvariantRecord:
    pair_id: str
    mechanism: str
    confounder: str
    seed: int
    permutation_only: bool
    same_nodes: bool
    same_static_edges: bool
    same_event_count: bool
    same_timestamp_multiset: bool
    same_degree_vector: bool
    same_confounder_events: bool
    same_static_view: bool
    positive_gold: bool
    negative_gold: bool
    positive_stat: float
    negative_stat: float
    positive_margin: float
    negative_margin: float
    required_margin: float
    global_oracle_check_negative: str      # "pass" | "fail: <reason>"
    rejection_attempts: int
    n_nodes: int
    n_events: int
    notes: str = ""

    @property
    def ok(self) -> bool:
        return all([
            self.permutation_only, self.same_nodes, self.same_static_edges,
            self.same_event_count, self.same_timestamp_multiset, self.same_degree_vector,
            self.same_confounder_events, self.same_static_view,
            self.positive_gold, not self.negative_gold,
            self.positive_margin >= self.required_margin,
            self.negative_margin >= self.required_margin,
            self.global_oracle_check_negative == "pass",
        ])

    def failures(self) -> list[str]:
        out = []
        for name in ("permutation_only", "same_nodes", "same_static_edges", "same_event_count",
                     "same_timestamp_multiset", "same_degree_vector", "same_confounder_events",
                     "same_static_view", "positive_gold"):
            if not getattr(self, name):
                out.append(name)
        if self.negative_gold:
            out.append("negative_gold_should_be_false")
        if self.positive_margin < self.required_margin:
            out.append(f"positive_margin {self.positive_margin:.4f} < {self.required_margin:.4f}")
        if self.negative_margin < self.required_margin:
            out.append(f"negative_margin {self.negative_margin:.4f} < {self.required_margin:.4f}")
        if self.global_oracle_check_negative != "pass":
            out.append(f"global_oracle:{self.global_oracle_check_negative}")
        return out

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ok"] = self.ok
        d["failures"] = self.failures()
        return d


def validate_invariant_record(d: dict[str, Any]) -> None:
    require_keys(d, ("pair_id", "mechanism", "permutation_only", "same_nodes",
                     "same_static_edges", "same_event_count", "same_timestamp_multiset",
                     "positive_gold", "negative_gold", "global_oracle_check_negative"),
                 "invariants")
    check_type(d["permutation_only"], bool, "invariants.permutation_only")


# --------------------------------------------------------------------------------------
# the checker
# --------------------------------------------------------------------------------------

def static_view_fingerprint(world: TemporalWorld, focus_nodes: list[int],
                            window: tuple[int, int]) -> str:
    """Everything the order-blind (A1) surface can observe, serialised.

    Equality of this fingerprint between the two worlds is the paper's central construction claim,
    so it is checked for every pair rather than argued once. It is deliberately computed over the
    *focus window*, which is the scope the agent is told to analyse: matching only global
    marginals would not be enough, because a windowed query could still separate the worlds.
    """
    import json
    from tools import static_surface as S

    nodes = [n for n in focus_nodes if n is not None][:20] or [0]
    return json.dumps({
        "summary": S.graph_summary(world, window=window),
        "degree": S.node_degree(world, nodes, window=window),
        "triangles": S.triangle_count(world, nodes, window=window),
        "top_active": S.top_active_nodes(world, 25, window=window),
        "khop": S.khop_subgraph(world, nodes[0], k=2, window=window),
        "neighbors": [S.neighbors(world, n, window=window) for n in nodes[:8]],
    }, sort_keys=True)


def check_pair(pos: TemporalWorld, neg: TemporalWorld, *, pair_id: str, mechanism: str,
               confounder: str, seed: int, gold_fn: Callable[[TemporalWorld], tuple[bool, float]],
               required_margin: float, threshold: float,
               global_check_fn: Callable[[TemporalWorld], str] | None = None,
               static_fingerprint_fn: Callable[[TemporalWorld], str] | None = None,
               rejection_attempts: int = 0) -> InvariantRecord:
    """Verify a candidate pair. ``gold_fn`` returns (label, statistic) for a world.

    ``positive_margin`` is how far the positive statistic clears ``threshold``; ``negative_margin``
    is how far the negative statistic falls below it. Both must reach ``required_margin`` so that
    neither world sits at the decision boundary (R2).
    """
    pos_label, pos_stat = gold_fn(pos)
    neg_label, neg_stat = gold_fn(neg)

    global_status = "pass"
    if global_check_fn is not None:
        global_status = global_check_fn(neg)

    same_static_view = True
    if static_fingerprint_fn is not None:
        same_static_view = static_fingerprint_fn(pos) == static_fingerprint_fn(neg)

    return InvariantRecord(
        pair_id=pair_id, mechanism=mechanism, confounder=confounder, seed=seed,
        permutation_only=_is_timestamp_permutation(pos, neg),
        same_nodes=(pos.n_nodes == neg.n_nodes and pos.directed == neg.directed),
        same_static_edges=(pos.static_edge_multiset() == neg.static_edge_multiset()),
        same_event_count=(pos.n_events == neg.n_events),
        same_timestamp_multiset=(pos.timestamp_multiset() == neg.timestamp_multiset()),
        same_degree_vector=bool(np.array_equal(pos.degree_vector(), neg.degree_vector())),
        same_confounder_events=(_confounder_multiset(pos) == _confounder_multiset(neg)),
        same_static_view=same_static_view,
        positive_gold=bool(pos_label), negative_gold=bool(neg_label),
        positive_stat=float(pos_stat), negative_stat=float(neg_stat),
        positive_margin=float(pos_stat - threshold),
        negative_margin=float(threshold - neg_stat),
        required_margin=float(required_margin),
        global_oracle_check_negative=global_status,
        rejection_attempts=rejection_attempts,
        n_nodes=pos.n_nodes, n_events=pos.n_events,
    )


def _is_timestamp_permutation(pos: TemporalWorld, neg: TemporalWorld) -> bool:
    """True iff neg reuses pos's edge multiset and timestamp multiset exactly.

    This is the formal content of R1: the two worlds differ only in which timestamp is attached to
    which event, never in which events exist or which instants are occupied.
    """
    if pos.n_events != neg.n_events:
        return False
    if pos.static_edge_multiset() != neg.static_edge_multiset():
        return False
    if pos.timestamp_multiset() != neg.timestamp_multiset():
        return False
    return True


def _confounder_multiset(w: TemporalWorld) -> Counter:
    m = w.tag == TAG_CONFOUNDER
    return Counter(zip(w.src[m].tolist(), w.dst[m].tolist(), w.t[m].tolist()))


def assert_pair_ok(rec: InvariantRecord) -> None:
    if not rec.ok:
        raise SchemaError(f"pair {rec.pair_id} failed invariants: {rec.failures()}")
