"""Compiler round-trip + blinding test.

Verifies three things that the whole experiment depends on:

1.  A served world reloaded from disk reproduces the gold oracle's verdict exactly — otherwise the
    agent and the answer key would be computing over different data.
2.  Served artifacts contain no polarity, mechanism, or planted-structure information (R5).
3.  The static (A1) surface returns *identical* results on both members of a pair, while the
    temporal (A3) surface separates them. This is the paper's core claim about the construction,
    so it is asserted mechanically rather than argued.
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tasks.compiler import compile_pair, load_world
from tasks.schema import FORBIDDEN_SERVED_KEYS
from tools import static_surface as S
from tools.temporal_surface import temporal_reachability
from worlds.mechanisms.cascade import cascade_gold

cfg = yaml.safe_load((ROOT / "configs/mechanisms.yaml").read_text())
results: list[tuple[bool, str]] = []


def ok(name: str, cond: bool, detail: str = "") -> None:
    results.append((cond, name))
    print(f"[{'  ok  ' if cond else ' FAIL '}] {name}{('  ' + detail) if detail else ''}")


print("compiling 5 cascade pairs...\n")
compiled = [compile_pair(cfg, "cascade", seed, scale="dev") for seed in range(5)]

# ---------------------------------------------------------------- 1. oracle round-trip
print("--- oracle round-trip through served files ---")
for cp in compiled:
    focus = cp.positive.task.focus
    kw = dict(source=focus["source"], target=focus["target"],
              window=tuple(focus["window"]), max_wait=focus["max_wait"])
    pos_label, _ = cascade_gold(load_world(cp.positive.world_path), **kw)
    neg_label, _ = cascade_gold(load_world(cp.negative.world_path), **kw)
    ok(f"{cp.pair_id}: served worlds reproduce gold (+{pos_label} -{neg_label})",
       pos_label is True and neg_label is False)

# ---------------------------------------------------------------- 2. blinding
print("\n--- blinding of served artifacts ---")
cp = compiled[0]
served_task = cp.positive.task.to_dict()
blob = json.dumps(served_task).lower()
leaks = [k for k in FORBIDDEN_SERVED_KEYS if f'"{k}"' in blob]
ok("served task carries no forbidden keys", not leaks, str(leaks))
ok("served question never says positive/negative/planted",
   not any(w in served_task["question"].lower()
           for w in ("positive", "negative", "planted", "ground truth", "counterfactual")))
ok("task ids of the two polarities are unrelated strings",
   cp.positive.task.task_id != cp.negative.task.task_id)

csv_text = (ROOT / cp.positive.world_path).read_text()
ok("served CSV has exactly the u,v,t header", csv_text.splitlines()[0] == "u,v,t")
meta = json.loads((ROOT / cp.positive.world_path).with_suffix(".meta.json").read_text())
ok("sidecar exposes only domain facts", set(meta) == {"n_nodes", "horizon", "directed"},
   str(sorted(meta)))
meta_neg = json.loads((ROOT / cp.negative.world_path).with_suffix(".meta.json").read_text())
ok("sidecar is identical for both polarities", meta == meta_neg)

# ---------------------------------------------------------------- 3. the separation claim
print("\n--- static surface cannot separate the pair; temporal surface can ---")
n_static_same = n_temporal_diff = 0
for cp in compiled:
    focus = cp.positive.task.focus
    w_pos, w_neg = load_world(cp.positive.world_path), load_world(cp.negative.world_path)
    win = tuple(focus["window"])
    s, t = focus["source"], focus["target"]

    static_views = []
    for w in (w_pos, w_neg):
        static_views.append(json.dumps({
            "summary": S.graph_summary(w, window=win),
            "deg": S.node_degree(w, [s, t], window=win),
            "nbr_s": S.neighbors(w, s, window=win),
            "nbr_t": S.neighbors(w, t, window=win),
            "khop": S.khop_subgraph(w, s, k=2, window=win),
            "tri": S.triangle_count(w, [s, t], window=win),
            "top": S.top_active_nodes(w, 20, window=win),
            "between": S.edges_between(w, s, t, window=win),
        }, sort_keys=True))
    same_static = static_views[0] == static_views[1]
    n_static_same += same_static

    r_pos = temporal_reachability(w_pos, s, t, window=win, max_wait=focus["max_wait"])
    r_neg = temporal_reachability(w_neg, s, t, window=win, max_wait=focus["max_wait"])
    diff_temporal = r_pos.reached and not r_neg.reached
    n_temporal_diff += diff_temporal
    ok(f"{cp.pair_id}: static identical={same_static}, temporal separates={diff_temporal}",
       same_static and diff_temporal)

ok("ALL pairs: whole static surface identical across polarity", n_static_same == len(compiled))
ok("ALL pairs: temporal reachability separates polarity", n_temporal_diff == len(compiled))

n_fail = sum(1 for good, _ in results if not good)
print(f"\n{'=' * 62}\n{len(results) - n_fail}/{len(results)} checks passed")
if n_fail:
    print("FAILED: " + ", ".join(n for g, n in results if not g))
sys.exit(1 if n_fail else 0)
