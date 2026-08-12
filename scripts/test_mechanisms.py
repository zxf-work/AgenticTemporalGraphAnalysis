"""Acceptance test for all three mechanisms (tickets T1.3, T2.1, T2.2).

For each mechanism, asserts the same four properties:
  1. every generated pair passes the full invariant check
  2. the served worlds reproduce the gold labels
  3. the whole static surface is identical across the pair
  4. the mechanism's own temporal operator separates the pair

Run:  python scripts/test_mechanisms.py [n_seeds]
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tasks.compiler import compile_pair, load_world
from tools import static_surface as S
from tools.temporal_surface import (neighborhood_evolution, temporal_closure,
                                    temporal_reachability)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
cfg = yaml.safe_load((ROOT / "configs/mechanisms.yaml").read_text())
results: list[tuple[bool, str]] = []


def ok(name: str, cond: bool, detail: str = "") -> None:
    results.append((cond, name))
    print(f"[{'  ok  ' if cond else ' FAIL '}] {name}{('  ' + detail) if detail else ''}")


def static_fingerprint(w, focus_nodes, window) -> str:
    """Everything the A1 surface can see, serialised."""
    return json.dumps({
        "summary": S.graph_summary(w, window=window),
        "deg": S.node_degree(w, focus_nodes[:20], window=window),
        "khop": S.khop_subgraph(w, focus_nodes[0], k=2, window=window),
        "tri": S.triangle_count(w, focus_nodes[:20], window=window),
        "top": S.top_active_nodes(w, 25, window=window),
        "nbrs": [S.neighbors(w, n, window=window) for n in focus_nodes[:8]],
    }, sort_keys=True)


def temporal_separates(mech, w_pos, w_neg, focus) -> tuple[bool, str]:
    win = tuple(focus["window"])
    if mech == "cascade":
        a = temporal_reachability(w_pos, focus["source"], focus["target"], window=win,
                                  max_wait=focus["max_wait"]).reached
        b = temporal_reachability(w_neg, focus["source"], focus["target"], window=win,
                                  max_wait=focus["max_wait"]).reached
        return (a and not b), f"reached +{a} -{b}"
    if mech == "closure_wave":
        a = temporal_closure(w_pos, focus["nodes"], window=win, delta=focus["delta"]).fraction
        b = temporal_closure(w_neg, focus["nodes"], window=win, delta=focus["delta"]).fraction
        return (a > b), f"closed fraction +{a:.3f} -{b:.3f}"
    a = neighborhood_evolution(w_pos, focus["group"], window=win,
                               bandwidth=focus["bandwidth"]).peak_change
    b = neighborhood_evolution(w_neg, focus["group"], window=win,
                               bandwidth=focus["bandwidth"]).peak_change
    return (a > b), f"peak change +{a:.3f} -{b:.3f}"


for mech in ("cascade", "closure_wave", "regime_switch"):
    print(f"\n{'=' * 66}\n{mech}\n{'=' * 66}")
    t0 = time.time()
    pairs, errors = [], []
    for seed in range(N):
        try:
            pairs.append(compile_pair(cfg, mech, seed, scale="dev"))
        except RuntimeError as exc:
            errors.append((seed, str(exc)[:90]))
    elapsed = time.time() - t0

    ok(f"{mech}: all {N} seeds produced a pair", len(pairs) == N,
       f"failed={[e[0] for e in errors]}")
    if not pairs:
        continue
    print(f"       ({elapsed:.1f}s, {elapsed / max(1, len(pairs)):.2f}s/pair)")

    n_inv = sum(1 for p in pairs if p.invariants.ok)
    ok(f"{mech}: every pair passes invariants", n_inv == len(pairs),
       f"{n_inv}/{len(pairs)}; first failure: "
       f"{next((p.invariants.failures() for p in pairs if not p.invariants.ok), None)}")

    pos_m = np.array([p.invariants.positive_margin for p in pairs])
    neg_m = np.array([p.invariants.negative_margin for p in pairs])
    req = cfg[{"cascade": "m1_cascade", "closure_wave": "m2_closure_wave",
               "regime_switch": "m3_regime_switch"}[mech]]["required_margin"]
    print(f"       margins: pos min={pos_m.min():.3f} mean={pos_m.mean():.3f} | "
          f"neg min={neg_m.min():.3f} mean={neg_m.mean():.3f} (required {req})")

    n_static_same = n_temporal_diff = 0
    for cp in pairs:
        focus = cp.positive.task.focus
        w_pos, w_neg = load_world(cp.positive.world_path), load_world(cp.negative.world_path)
        fnodes = (focus.get("nodes") or focus.get("group")
                  or [focus.get("source"), focus.get("target")])
        win = tuple(focus["window"])
        n_static_same += static_fingerprint(w_pos, fnodes, win) == static_fingerprint(
            w_neg, fnodes, win)
        sep, _ = temporal_separates(mech, w_pos, w_neg, focus)
        n_temporal_diff += sep

    ok(f"{mech}: static surface identical across every pair", n_static_same == len(pairs),
       f"{n_static_same}/{len(pairs)}")
    ok(f"{mech}: temporal operator separates every pair", n_temporal_diff == len(pairs),
       f"{n_temporal_diff}/{len(pairs)}")

    ex = pairs[0]
    _, detail = temporal_separates(mech, load_world(ex.positive.world_path),
                                   load_world(ex.negative.world_path), ex.positive.task.focus)
    print(f"       example ({ex.pair_id}): {detail}")

n_fail = sum(1 for good, _ in results if not good)
print(f"\n{'=' * 66}\n{len(results) - n_fail}/{len(results)} checks passed")
if n_fail:
    print("FAILED: " + ", ".join(n for g, n in results if not g))
sys.exit(1 if n_fail else 0)
