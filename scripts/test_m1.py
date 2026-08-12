"""T1.3 acceptance test — M1 cascade pair generator.

Required (taskboard Phase 1 / plan WP1):
  * N generated pairs, 100% pass invariant checks
  * 100% have differing gold mechanism predicate
  * re-running a seed produces byte-identical world hashes

Run:  python scripts/test_m1.py [n_pairs]
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from worlds.invariants import check_pair
from worlds.mechanisms.cascade import (cascade_global_check, cascade_gold, generate_cascade_pair)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
cfg = yaml.safe_load(open(pathlib.Path(__file__).resolve().parents[1] / "configs/mechanisms.yaml"))
m_cfg = cfg["m1_cascade"]

print(f"generating {N} cascade pairs (dev scale)...")
records, attempts_hist, t0 = [], [], time.time()
failures = []

for seed in range(N):
    pos, neg, spec, attempts = generate_cascade_pair(cfg, seed, scale="dev")
    attempts_hist.append(attempts)
    gold_kw = dict(source=spec.source, target=spec.target, window=spec.window,
                   max_wait=spec.max_wait)
    rec = check_pair(
        pos, neg, pair_id=f"m1_none_seed{seed:03d}", mechanism="cascade", confounder="none",
        seed=seed, gold_fn=lambda w: cascade_gold(w, **gold_kw),
        required_margin=m_cfg["required_margin"], threshold=m_cfg["threshold"],
        global_check_fn=lambda w: cascade_global_check(w, **gold_kw),
        rejection_attempts=attempts)
    records.append(rec)
    if not rec.ok:
        failures.append((seed, rec.failures()))

elapsed = time.time() - t0
n_ok = sum(1 for r in records if r.ok)
print(f"\n{n_ok}/{N} pairs pass all invariants   ({elapsed:.1f}s, "
      f"{elapsed / N:.2f}s/pair)")

if failures:
    print("\nFAILURES:")
    for seed, f in failures[:10]:
        print(f"  seed {seed}: {f}")

# ---------------------------------------------------------------- invariant breakdown
print("\ninvariant breakdown (all must be N/N):")
for field in ("permutation_only", "same_nodes", "same_static_edges", "same_event_count",
              "same_timestamp_multiset", "same_degree_vector", "same_confounder_events",
              "positive_gold"):
    k = sum(1 for r in records if getattr(r, field))
    print(f"  {field:26s} {k}/{N}")
k = sum(1 for r in records if not r.negative_gold)
print(f"  {'negative_gold is False':26s} {k}/{N}")
k = sum(1 for r in records if r.global_oracle_check_negative == 'pass')
print(f"  {'negative globally clean':26s} {k}/{N}")

# ---------------------------------------------------------------- margins
pos_m = np.array([r.positive_margin for r in records])
neg_m = np.array([r.negative_margin for r in records])
print(f"\nmargins (required >= {m_cfg['required_margin']}):")
print(f"  positive  min={pos_m.min():.3f}  mean={pos_m.mean():.3f}  max={pos_m.max():.3f}")
print(f"  negative  min={neg_m.min():.3f}  mean={neg_m.mean():.3f}  max={neg_m.max():.3f}")
print(f"\nrejection attempts: mean={np.mean(attempts_hist):.2f}  max={max(attempts_hist)}")
print(f"world size: {records[0].n_nodes} nodes, {records[0].n_events} events")

# ---------------------------------------------------------------- determinism
print("\ndeterminism check (regenerate seeds 0,1,2):")
det_ok = True
for seed in (0, 1, 2):
    p1, n1, s1, _ = generate_cascade_pair(cfg, seed, scale="dev")
    p2, n2, s2, _ = generate_cascade_pair(cfg, seed, scale="dev")
    same = (p1.world_hash() == p2.world_hash() and n1.world_hash() == n2.world_hash()
            and s1.path_nodes == s2.path_nodes and s1.times == s2.times)
    det_ok &= same
    print(f"  seed {seed}: {'identical' if same else 'MISMATCH'}  "
          f"(+{p1.world_hash()[:10]} / -{n1.world_hash()[:10]})")

# the two worlds must differ from each other, or the probe is vacuous
distinct = all(
    generate_cascade_pair(cfg, s, scale="dev")[0].world_hash()
    != generate_cascade_pair(cfg, s, scale="dev")[1].world_hash() for s in (0, 1, 2))
print(f"  positive != negative hash: {distinct}")

ok = (n_ok == N) and det_ok and distinct
print(f"\n{'=' * 62}\nT1.3 ACCEPTANCE: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
