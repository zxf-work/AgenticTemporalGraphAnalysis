"""Regression test: worlds generated with different parameters must not collide.

A sweep over a generator axis writes one world per (mechanism, confounder, seed, parameters). If
the filename ignores the parameters, later variants overwrite earlier ones and every cell reads
whichever was compiled last -- the axis then appears to have no effect, which is exactly the
silent failure that invalidated the first ambiguity sweep.
"""
from __future__ import annotations
import pathlib, sys
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.runner import Cell, MatrixRunner
from tasks.compiler import load_world

cfg = yaml.safe_load((ROOT / "configs/mechanisms.yaml").read_text())
results = []
def ok(name, cond, detail=""):
    results.append((cond, name))
    print(f"[{'  ok  ' if cond else ' FAIL '}] {name}{('  ' + detail) if detail else ''}")

cells = [Cell(mechanism="cascade", seed=0, confounder="none", condition="A1",
              model_slot="script_a1", budget=8,
              overrides=(("hops", 5), ("n_decoy_paths", d), ("temporal_resolution", 512)))
         for d in (0, 4, 16, 64)]

runner = MatrixRunner(pathlib.Path("/tmp/iso_test"), backends={}, concurrency=1, mech_cfg=cfg,
                      verbose=False)
pairs = runner.prepare_worlds(cells)
runner.close()

ok(f"one world per D value ({len(pairs)}/4)", len(pairs) == 4)

paths = {c.world_key: pairs[c.world_key][0].world_path for c in cells if c.world_key in pairs}
ok("world paths are distinct", len(set(paths.values())) == len(paths),
   f"{len(set(paths.values()))} distinct of {len(paths)}")

hashes = {k: pairs[k][0].world_hash for k in pairs}
ok("world hashes are distinct", len(set(hashes.values())) == len(hashes))

# the D axis must actually change the graph the agent sees
sizes = {}
for c in cells:
    if c.world_key not in pairs:
        continue
    d = dict(c.overrides)["n_decoy_paths"]
    sizes[d] = load_world(pairs[c.world_key][0].world_path).n_events
print(f"       events by D: {sizes}")
ok("more decoys means more events", all(sizes[a] < sizes[b] for a, b in ((0,4),(4,16),(16,64))),
   str(sizes))

n_fail = sum(1 for g, _ in results if not g)
print(f"\n{len(results)-n_fail}/{len(results)} checks passed")
sys.exit(1 if n_fail else 0)
