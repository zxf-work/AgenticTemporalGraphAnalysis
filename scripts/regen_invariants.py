"""Recompile the worlds a completed sweep used, to restore invariant records.

World generation is deterministic in (seed, config), so the pairs an agent actually saw can be
reconstructed exactly and re-verified without re-running a single episode. Used after the
invariant artifacts were lost to an over-broad rsync --delete.

Verifies the reconstruction rather than assuming it: every regenerated world hash must match the
hash recorded in the sweep's results.
"""
from __future__ import annotations
import copy, json, pathlib, sys
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tasks.compiler import compile_pair
from common import stable_hash

RUN = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "runs/ambiguity")
manifest = json.loads((RUN / "manifest.json").read_text())
cfg = yaml.safe_load((ROOT / "configs/mechanisms.yaml").read_text())

seen, n_ok, mismatches = set(), 0, []
recorded = {}
for line in (RUN / "results.jsonl").read_text().splitlines():
    if line.strip():
        d = json.loads(line)
        recorded[(d["pair_id"], d["polarity"])] = d.get("world_hash")

for c in manifest["cells"]:
    key = (c["mechanism"], c["seed"], c["confounder"], c["paraphrase"], c["scale"],
           tuple(map(tuple, c["overrides"])))
    if key in seen:
        continue
    seen.add(key)
    cfg2 = copy.deepcopy(cfg)
    mk = {"cascade": "m1_cascade", "closure_wave": "m2_closure_wave",
          "regime_switch": "m3_regime_switch"}[c["mechanism"]]
    for k, v in c["overrides"]:
        (cfg2[mk] if k in cfg2[mk] else cfg2["background"])[k] = v
    variant = stable_hash(sorted(map(tuple, c["overrides"])), length=6) if c["overrides"] else ""
    cp = compile_pair(cfg2, c["mechanism"], c["seed"], confounder=c["confounder"],
                      scale=c["scale"], paraphrase=c["paraphrase"], variant=variant)
    n_ok += cp.invariants.ok
    for rec in (cp.positive, cp.negative):
        want = recorded.get((cp.pair_id, rec.polarity))
        if want and want != rec.world_hash:
            mismatches.append((cp.pair_id, rec.polarity, want, rec.world_hash))

print(f"regenerated {len(seen)} pairs; {n_ok}/{len(seen)} pass invariants")
print(f"world-hash mismatches vs the sweep's records: {len(mismatches)}")
for m in mismatches[:5]:
    print("  ", m)
sys.exit(1 if (mismatches or n_ok != len(seen)) else 0)
