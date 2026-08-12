"""Recover the compiled pair a scored row came from.

Shared by `scripts/rescore.py` and `scripts/replay_backend.py`. Compilation is deterministic and
free, so any analysis that needs the task record or the world behind a completed run can rebuild
it exactly rather than storing a second copy beside the trace.

The sweep name carries the axis value (``resolution=512``) and the declared ``hold`` block carries
the rest, which together reproduce the generator overrides — and therefore the ``variant`` hash
that keeps worlds differing only in generator parameters from sharing a filename.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import stable_hash
# MECH_KEY is imported, never restated. Three separate copies of this map existed at one point and
# each had to learn about M4 independently; the one here silently skipped every compound row.
from tasks.compiler import MECH_KEY, compile_pair


def overrides_for(row: dict[str, Any], xcfg: dict[str, Any]) -> dict[str, Any]:
    sweep = (row.get("sweep") or "").split("=")
    name, value = (sweep[0], sweep[1]) if len(sweep) == 2 else ("", "")
    spec = xcfg["sweeps"].get(name, {})
    ov = dict(spec.get("hold", {}))
    axis = spec.get("axis")
    if axis and axis != "mechanism" and value != "":
        try:
            ov[axis] = int(value)
        except ValueError:
            ov[axis] = value
    return ov


def compiled_pair(row: dict[str, Any], mcfg: dict[str, Any], xcfg: dict[str, Any],
                  cache: dict[tuple, Any], *, write: bool = False):
    mech = row["mechanism"]
    seed = int(row.get("seed", 0))
    conf = row.get("confounder", "none")
    ov = overrides_for(row, xcfg)
    key = (mech, seed, conf, tuple(sorted(ov.items())))
    if key not in cache:
        variant = stable_hash(sorted(ov.items()), length=6) if ov else ""
        cfg = json.loads(json.dumps(mcfg))
        mk = MECH_KEY[mech]
        for k, v in ov.items():
            if k in cfg[mk]:
                cfg[mk][k] = v
            elif k in cfg["background"]:
                cfg["background"][k] = v
        cache[key] = compile_pair(cfg, mech, seed, confounder=conf, scale="dev", paraphrase=1,
                                  write=write, variant=variant)
    return cache[key]


def record_for(row: dict[str, Any], mcfg: dict[str, Any], xcfg: dict[str, Any],
               cache: dict[tuple, Any], *, write: bool = False):
    """The TaskRecord for this row's polarity."""
    cp = compiled_pair(row, mcfg, xcfg, cache, write=write)
    return getattr(cp, row["polarity"])
