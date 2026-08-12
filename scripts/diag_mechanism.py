"""Print the raw gold statistics for one mechanism, so thresholds are set from evidence.

Bypasses the admissibility filter and reports what the positive and negative worlds actually
score, which is what you need to see when a generator rejects every seed.

Run:  python scripts/diag_mechanism.py <cascade|closure_wave|regime_switch> [n_seeds]
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MECH = sys.argv[1] if len(sys.argv) > 1 else "closure_wave"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 5
cfg = yaml.safe_load((ROOT / "configs/mechanisms.yaml").read_text())

from worlds.background import low_traffic_nodes, make_background
from worlds.base import TAG_MECHANISM, TemporalWorld
from tools.temporal_surface import neighborhood_evolution, temporal_closure

bg = cfg["background"]
horizon = tuple(bg["horizon"])


def plant(base, extra):
    src = np.concatenate([base.src, np.array([e[0] for e in extra], dtype=np.int32)])
    dst = np.concatenate([base.dst, np.array([e[1] for e in extra], dtype=np.int32)])
    t = np.concatenate([base.t, np.array([e[2] for e in extra], dtype=np.int64)])
    tag = np.concatenate([base.tag, np.array([e[3] for e in extra], dtype=np.uint8)])
    return TemporalWorld(base.n_nodes, src, dst, t, tag, base.horizon, base.directed, {})


if MECH == "closure_wave":
    m = cfg["m2_closure_wave"]
    print(f"threshold={m['threshold']} margin={m['required_margin']} delta={m['delta']}")
    print(f"  need: positive >= {m['threshold'] + m['required_margin']:.2f}, "
          f"negative <= {m['threshold'] - m['required_margin']:.2f}\n")
    for seed in range(N):
        sub = np.random.default_rng(seed * 1000 + 1)
        base = make_background(bg["n_nodes_dev"], bg["events_per_node"], horizon, rng=sub,
                               rate_alpha=bg["rate_alpha"], burstiness=bg["burstiness"])
        picked = low_traffic_nodes(base, m["n_wedges"] * 3, rng=sub)
        tris = [tuple(picked[3 * i:3 * i + 3]) for i in range(m["n_wedges"])]
        nodes = sorted(picked)
        centre = float(sub.uniform(horizon[0] + 0.25 * horizon[1], horizon[1] * 0.75))
        win = (int(centre - m["wave_window"]), int(centre + m["wave_window"]))

        tri_edges = [[(a, b), (b, c), (a, c)] for a, b, c in tris]
        pos_times = []
        for _ in tris:
            t1 = int(sub.uniform(win[0], win[1] - m["delta"]))
            t2 = int(t1 + sub.uniform(1, m["delta"]))
            t3 = int(t2 + sub.uniform(1, m["delta"] * 0.8))
            pos_times.append([t1, t2, t3])
        pos_ev = [(u, v, t, TAG_MECHANISM) for es, ts in zip(tri_edges, pos_times)
                  for (u, v), t in zip(es, ts)]
        flat = sorted(t for ts in pos_times for t in ts)
        n_tri = len(tris)
        neg_ev = [(u, v, flat[j * n_tri + i], TAG_MECHANISM)
                  for i, es in enumerate(tri_edges) for j, (u, v) in enumerate(es)]

        p = temporal_closure(plant(base, pos_ev), nodes, window=win, delta=m["delta"])
        n = temporal_closure(plant(base, neg_ev), nodes, window=win, delta=m["delta"])
        print(f"seed {seed}: POS frac={p.fraction:.3f} ({p.n_closed_within_delta}/{p.n_triangles}"
              f", median_lag={p.median_closing_lag:.0f}) | "
              f"NEG frac={n.fraction:.3f} ({n.n_closed_within_delta}/{n.n_triangles}"
              f", median_lag={n.median_closing_lag:.0f})")

elif MECH == "regime_switch":
    m = cfg["m3_regime_switch"]
    print(f"threshold={m['threshold']} margin={m['required_margin']}")
    print(f"  need: positive >= {m['threshold'] + m['required_margin']:.2f}, "
          f"negative <= {m['threshold'] - m['required_margin']:.2f}\n")
    for seed in range(N):
        sub = np.random.default_rng(seed * 1000 + 1)
        base = make_background(bg["n_nodes_dev"], bg["events_per_node"], horizon, rng=sub,
                               rate_alpha=bg["rate_alpha"], burstiness=bg["burstiness"])
        picked = low_traffic_nodes(base, m["group_size"] + m["old_partners"] + m["new_partners"],
                                   rng=sub)
        g = sorted(picked[:m["group_size"]])
        old = sorted(picked[m["group_size"]:m["group_size"] + m["old_partners"]])
        new = sorted(picked[m["group_size"] + m["old_partners"]:])
        bw = m["bandwidth"]
        span = bw * 2
        tc = int(sub.uniform(horizon[0] + span + bw, horizon[1] - span - bw))
        win = (max(horizon[0], tc - span - bw), min(horizon[1], tc + span + bw))
        oe = [(x, p) for x in g for p in old]
        ne = [(x, p) for x in g for p in new]
        tb = sub.uniform(win[0], tc - m["transition_window"] / 2, size=len(oe))
        ta = sub.uniform(tc + m["transition_window"] / 2, win[1], size=len(ne))
        times = np.concatenate([tb, ta]).round().astype(int)
        edges = oe + ne
        pos_ev = [(u, v, int(t), TAG_MECHANISM) for (u, v), t in zip(edges, times)]

        byp: dict[int, list] = {}
        for (a, b) in edges:
            byp.setdefault(b, []).append((a, b))
        parts = sorted(byp)
        inter = []
        for i in range(max(len(v) for v in byp.values())):
            for p_ in parts:
                if i < len(byp[p_]):
                    inter.append(byp[p_][i])
        neg_ev = [(u, v, int(t), TAG_MECHANISM) for (u, v), t in zip(inter, sorted(times))]

        rp = neighborhood_evolution(plant(base, pos_ev), g, window=win, bandwidth=bw)
        rn = neighborhood_evolution(plant(base, neg_ev), g, window=win, bandwidth=bw)
        print(f"seed {seed}: POS peak={rp.peak_change:.3f} @t={rp.peak_time} "
              f"(|before|={len(rp.before)} |after|={len(rp.after)}) | "
              f"NEG peak={rn.peak_change:.3f} @t={rn.peak_time} "
              f"(|before|={len(rn.before)} |after|={len(rn.after)})")
else:
    print("cascade diagnostics: see scripts/test_m1.py")
