"""Re-score completed runs from their stored traces, without re-running any episode.

    python scripts/rescore.py runs/a2_mini runs/xmech_opus ...
    python scripts/rescore.py --all

Milestone C2 asks for EVPairAcc to be *recomputed* over work already done, and any later change
to a checker rule has the same need. Traces hold the full trajectory and the final claim, so every
metric downstream of them can be rebuilt for free; only the scoring is redone, never the episode.

`results.jsonl` is rewritten in place after the new rows are computed successfully, and the prior
file is kept as `results.jsonl.bak-<n>`. The scored row is a *derived* artifact — the trace is the
evidence — so regenerating it is safe in a way that re-running an episode never is.

The task record for each trace is recovered by recompiling its pair, which is deterministic and
costs no API calls. A trace whose world no longer compiles to the same hash is skipped and
reported rather than silently rescored against a different world.
"""

from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import asdict

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.score_run import score
from tasks.lookup import record_for
from traces.schema import Trace

args = [a for a in sys.argv[1:] if not a.startswith("--")]
if "--all" in sys.argv or not args:
    dirs = sorted(d for d in (ROOT / "runs").glob("*") if (d / "results.jsonl").exists())
else:
    dirs = [pathlib.Path(a) for a in args]

mcfg = yaml.safe_load((ROOT / "configs/mechanisms.yaml").read_text())
xcfg = yaml.safe_load((ROOT / "configs/experiment_matrix.yaml").read_text())
_pair_cache: dict[tuple, object] = {}


for d in dirs:
    src = d / "results.jsonl"
    rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    out, skipped, changed = [], 0, 0
    for row in rows:
        tp = d / "traces" / f"{row.get('run_id','')}.json"
        if not tp.exists():
            out.append(row)
            skipped += 1
            continue
        try:
            trace = Trace.from_dict(json.loads(tp.read_text()))
            rec = record_for(row, mcfg, xcfg, _pair_cache)
            if rec.world_hash != trace.world_hash:
                out.append(row)
                skipped += 1
                continue
            new = score(trace, rec, row["condition"])
        except Exception as exc:                                   # noqa: BLE001
            print(f"  ! {d.name}/{row.get('run_id')}: {type(exc).__name__}: {str(exc)[:90]}")
            out.append(row)
            skipped += 1
            continue
        payload = {**{k: v for k, v in asdict(new).items() if k != "check"},
                   "check": asdict(new.check)}
        merged = {**row, **payload}
        if merged != row:
            changed += 1
        out.append(merged)

    n = 0
    while (d / f"results.jsonl.bak-{n}").exists():
        n += 1
    (d / f"results.jsonl.bak-{n}").write_text(src.read_text())
    src.write_text("".join(json.dumps(r) + "\n" for r in out))
    print(f"{d.name}: {len(out)} rows, {changed} changed, {skipped} skipped "
          f"(backup results.jsonl.bak-{n})")
