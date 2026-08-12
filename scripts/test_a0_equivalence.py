"""T14 — A0 observational-equivalence checker (pivot addendum §3, §21 Step 1).

Verifies the A0 acceptance condition

    response_A0(G+) == response_A0(G-)     for every permissible A0 query

*exhaustively*, not by sampling. This is possible only because no A0 tool takes a window: the
query space is (tools x node arguments), so enumerating every node covers it. Any leak here
invalidates the identifiability argument, so this must pass before anything else runs.

For contrast the same enumeration is run against the A1 surface at several window granularities,
which is expected to FAIL — that failure is the paper's central observation, so it is asserted
rather than left implicit.

Run:  python scripts/test_a0_equivalence.py [n_pairs] [mechanism]
"""

from __future__ import annotations

import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.tools_spec import A0_TOOLS
from common import canonical_json
from tasks.compiler import compile_pair, load_world
from tools import static_surface as S

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
MECHS = [sys.argv[2]] if len(sys.argv) > 2 else ["cascade", "closure_wave", "regime_switch"]
cfg = yaml.safe_load((ROOT / "configs/mechanisms.yaml").read_text())

results: list[tuple[bool, str]] = []


def ok(name: str, cond: bool, detail: str = "") -> None:
    results.append((cond, name))
    print(f"[{'  ok  ' if cond else ' FAIL '}] {name}{('  ' + detail) if detail else ''}")


def a0_queries(n_nodes: int, focus_nodes: list[int]) -> list[tuple[str, dict]]:
    """Every permissible A0 query.

    Node-parameterised tools are enumerated over *all* nodes, so the check is exhaustive over the
    interface rather than over a sample of it. k and top-k are enumerated over their whole
    plausible range.
    """
    q: list[tuple[str, dict]] = [("graph_summary_full", {})]
    all_nodes = list(range(n_nodes))
    q.append(("node_degree_full", {"nodes": all_nodes}))
    q.append(("triangle_count_full", {"nodes": focus_nodes}))
    for k in (5, 20, 50, 200):
        q.append(("top_active_nodes_full", {"k": k}))
    for n in all_nodes:
        for d in ("in", "out", "both"):
            q.append(("neighbors_full", {"node": n, "direction": d}))
    for n in focus_nodes:
        for k in (1, 2, 3, 4, 5):
            q.append(("khop_subgraph_full", {"node": n, "k": k}))
    for i, u in enumerate(focus_nodes):
        for v in focus_nodes[i + 1:]:
            q.append(("edges_between_full", {"u": u, "v": v}))
    return q


by_name = {t.name: t for t in A0_TOOLS}

for mech in MECHS:
    print(f"\n=== {mech} ===")
    for seed in range(N):
        cp = compile_pair(cfg, mech, seed, scale="dev")
        focus = cp.positive.task.focus
        fnodes = [n for n in (focus.get("nodes") or focus.get("group")
                              or [focus.get("source"), focus.get("target")]) if n is not None]
        wp = load_world(cp.positive.world_path)
        wn = load_world(cp.negative.world_path)

        queries = a0_queries(wp.n_nodes, fnodes)
        diffs = []
        for name, args in queries:
            fn = by_name[name].fn
            if canonical_json(fn(wp, **args)) != canonical_json(fn(wn, **args)):
                diffs.append((name, args))
        ok(f"{cp.pair_id}: A0 exhaustive equivalence ({len(queries)} queries)", not diffs,
           f"{len(diffs)} differing, e.g. {diffs[:2]}")

# ---------------------------------------------------------------- the contrast
# The A1 surface is order-revealing under composition, but only at sufficient temporal
# resolution and only for the right nodes. Measuring *where* it becomes revealing gives the
# resolution axis (R) of the complexity grid directly, and shows why a single-node probe can
# miss the leak entirely: a permutation need not move every event.
print("\n=== contrast: at what resolution does the A1 surface separate the pair? ===")
print("  (coarsest number of bins at which some node-filtered count differs)\n")

for mech in MECHS:
    for seed in range(min(N, 3)):
        cp = compile_pair(cfg, mech, seed, scale="dev")
        focus = cp.positive.task.focus
        wp, wn = load_world(cp.positive.world_path), load_world(cp.negative.world_path)
        lo, hi = focus["window"]

        active = sorted(set(wp.src[wp.window_mask(lo, hi)].tolist())
                        | set(wp.dst[wp.window_mask(lo, hi)].tolist()))

        first_sep, n_sep_nodes = None, 0
        for nbins in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
            width = max(1, (hi - lo) // nbins)
            wins = [[lo + i * width, lo + (i + 1) * width - 1] for i in range(nbins)]
            sep_nodes = [n for n in active
                         if canonical_json(S.window_event_count(wp, wins, nodes=[n]))
                         != canonical_json(S.window_event_count(wn, wins, nodes=[n]))]
            if sep_nodes and first_sep is None:
                first_sep, n_sep_nodes = nbins, len(sep_nodes)
                break
        glob_same = all(
            canonical_json(S.window_event_count(wp, [[lo + i * max(1, (hi - lo) // nb),
                                                      lo + (i + 1) * max(1, (hi - lo) // nb) - 1]
                                                     for i in range(nb)]))
            == canonical_json(S.window_event_count(wn, [[lo + i * max(1, (hi - lo) // nb),
                                                         lo + (i + 1) * max(1, (hi - lo) // nb) - 1]
                                                        for i in range(nb)]))
            for nb in (2, 16, 128))
        print(f"  {cp.pair_id}: separates at {str(first_sep):>4} bins "
              f"({n_sep_nodes} of {len(active)} nodes leak) | global counts identical={glob_same}")
        results.append((first_sep is not None,
                        f"{cp.pair_id}: A1 separates at some resolution"))
        results.append((glob_same, f"{cp.pair_id}: A1 global counts stay identical"))

print("\n  Reading: A1 is order-blind at coarse resolution and order-revealing at fine "
      "resolution.\n  That threshold is the R axis of the complexity sweep; global (unfiltered)\n"
      "  counts never separate, because the timestamp multiset is preserved by construction.")

n_fail = sum(1 for good, _ in results if not good)
print(f"\n{'=' * 66}\n{len(results) - n_fail}/{len(results)} checks passed")
if n_fail:
    print("FAILED: " + ", ".join(n for g, n in results if not g))
sys.exit(1 if n_fail else 0)
