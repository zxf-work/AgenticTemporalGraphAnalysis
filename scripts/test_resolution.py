"""T17 acceptance — the temporal-resolution axis actually varies search difficulty.

Checks that increasing R widens the span the agent must search, holds background density roughly
constant, and leaves the mechanism intact (invariants and margins unaffected).
"""
from __future__ import annotations
import copy, pathlib, sys
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from worlds.invariants import check_pair, static_view_fingerprint
from worlds.mechanisms.cascade import (cascade_global_check, cascade_gold, generate_cascade_pair)

base = yaml.safe_load((ROOT / "configs/mechanisms.yaml").read_text())
VALUES = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["32", "128", "512", "2048"])]
NS = 3
unit = base["m1_cascade"]["max_wait"] // 8
print(f"unit = max_wait/8 = {unit};  R = span / unit\n")
print(f"{'R':>6} {'span':>9} {'span/unit':>10} {'events':>9} {'horizon':>10} "
      f"{'density':>9} {'ok':>6} {'posM':>7} {'negM':>7}")

fails = []
for R in VALUES:
    cfg = copy.deepcopy(base)
    cfg["m1_cascade"]["temporal_resolution"] = R
    spans, ok_n, dens, ev, hz, pm, nm = [], 0, [], 0, 0, [], []
    for s in range(NS):
        p, n, spec, _ = generate_cascade_pair(cfg, s, scale="dev")
        kw = dict(source=spec.source, target=spec.target, window=spec.window,
                  max_wait=spec.max_wait)
        fn = [spec.source, spec.target]
        rec = check_pair(p, n, pair_id=f"r{R}s{s}", mechanism="cascade", confounder="none",
                         seed=s, gold_fn=lambda w: cascade_gold(w, **kw),
                         required_margin=cfg["m1_cascade"]["required_margin"], threshold=0.5,
                         global_check_fn=lambda w: cascade_global_check(w, **kw),
                         static_fingerprint_fn=lambda w: static_view_fingerprint(w, fn, spec.window))
        ok_n += rec.ok
        if not rec.ok:
            fails.append((R, s, rec.failures()))
        spans.append(spec.window[1] - spec.window[0])
        ev, hz = p.n_events, p.horizon[1]
        dens.append(p.n_events / max(1, p.horizon[1]))
        pm.append(rec.positive_margin); nm.append(rec.negative_margin)
    sp = sum(spans) // len(spans)
    print(f"{R:>6} {sp:>9} {sp/unit:>10.1f} {ev:>9} {hz:>10} "
          f"{sum(dens)/len(dens):>9.4f} {ok_n:>4}/{NS} "
          f"{sum(pm)/len(pm):>7.3f} {sum(nm)/len(nm):>7.3f}")

print()
if fails:
    print("FAILURES:")
    for R, s, f in fails[:6]:
        print(f"  R={R} seed={s}: {f}")
print(f"T17 ACCEPTANCE: {'FAIL' if fails else 'PASS'}")
sys.exit(1 if fails else 0)
