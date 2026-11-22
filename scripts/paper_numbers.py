"""Every number quoted in the paper, emitted from the run data in one place.

    python scripts/paper_numbers.py                        # the paper's numbers
    python scripts/paper_numbers.py --ev-rule submitted    # as in the reviewed submission

Evidence validity (every `ev=` / `citedEV=` value below) follows the rule the paper states: at
least one cited support step satisfies the task-constraint test. The reviewed submission's values
were computed with a stricter rule for A3 on M1-M3; `--ev-rule submitted` reproduces them
(`evaluation/ev_rule.py`). No other quantity depends on the rule.

Written so that no figure in the manuscript is transcribed by hand. Building Figure 1 this way
already caught a per-seed transcription error in `PAPER_CLAIMS.md`; the same discipline applies to
the paper text. If a number appears in the LaTeX and not here, it is unverified.
"""

from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.ev_rule import ev_rule_from_argv
from evaluation.figdata import MODELS, by_interface, ev_valid, load, paired, resolution_axis

EV_RULE = ev_rule_from_argv(sys.argv)
rows = load()
LADDER = {"qwen2.5-7b-instruct-awq": "Qwen2.5-7B", "qwen2.5-14b-instruct-awq": "Qwen2.5-14B",
          "qwen2.5-32b-instruct-awq": "Qwen2.5-32B"}


def rule(s):
    print(f"\n{'=' * 78}\n{s}\n{'=' * 78}")


rule("0. CORPUS")
print(f"evidence-validity rule   : {EV_RULE}"
      + ("  (the reviewed submission's rule; the paper's is `stated`)"
         if EV_RULE != "stated" else ""))
print(f"valid scored runs        : {len(rows)}")
print(f"mechanisms               : {dict(Counter(r['mechanism'] for r in rows))}")
print(f"interfaces               : {dict(Counter(r['condition'] for r in rows))}")
print(f"models                   : {dict(Counter(r['model'] for r in rows))}")
print(f"scale                    : {dict(Counter(r.get('scale') for r in rows))}")
print(f"confounders              : {dict(Counter(r.get('confounder') for r in rows))}")
spend = defaultdict(float)
for r in rows:
    spend[r["model"]] += r.get("cost_usd", 0)
print("spend (valid runs only)  : " + ", ".join(f"{k} ${v:.2f}" for k, v in spend.items() if v))

rule("1. M1 RESOLUTION AXIS (B=16, hops=5, D=0)")
A = resolution_axis(rows)
RS = [32, 128, 192, 256, 384, 512]
print(f"{'cond':<5}{'model':<14}" + "".join(f"{R:>9}" for R in RS))
for cond in ("A1", "A2", "A3"):
    for m in MODELS:
        d = A.get((cond, m), {})
        if d:
            print(f"{cond:<5}{MODELS[m]:<14}"
                  + "".join(f"{d[R][0]:>9.2f}" if R in d else f"{'-':>9}" for R in RS))
print("\nOpus A1 per-seed and reconstruction rate:")
recon = defaultdict(list)
for r in rows:
    name, _, val = (r.get("sweep") or "").partition("=")
    # baseline condition only -- the robustness cells reuse the `resolution` sweep name at
    # R=128, and pair_id does not encode scale, so they merge into this cell otherwise
    if (name in ("resolution", "resolution_fine") and r["condition"] == "A1"
            and r["model"] == "claude-opus-5" and r.get("budget") == 16
            and r.get("confounder", "none") == "none" and r.get("scale", "dev") == "dev"):
        recon[int(val)].append(bool(r.get("used_reconstruction")))
d = A[("A1", "claude-opus-5")]
prev = None
for R in RS:
    ok = {i for i, b in enumerate(d[R][3]) if b == "1"}
    nest = "" if prev is None else ("nested" if ok <= prev else "NOT NESTED")
    rr = sum(recon[R]) / len(recon[R]) if recon.get(R) else float("nan")
    print(f"  R={R:<5} {d[R][3]}  acc={d[R][0]:.2f}  recon={rr:.2f}  {nest}")
    prev = ok
print("mini A1 reconstruction rate (all R pooled):", end=" ")
mr = [bool(r.get("used_reconstruction")) for r in rows
      if (r.get("sweep") or "").startswith(("resolution=", "resolution_fine="))
      and r["condition"] == "A1" and r["model"] == "gpt-5-mini"
      and r.get("confounder", "none") == "none" and r.get("scale", "dev") == "dev"]
print(f"{sum(mr) / len(mr):.2f}" if mr else "n/a")

rule("2. M2 CLOSURE WAVE / M3 REGIME SWITCH (budget axis)")
for mech in ("closure_wave", "regime_switch"):
    print(f"-- {mech}")
    g = defaultdict(list)
    for r in rows:
        if r["mechanism"] == mech and r["model"] in MODELS:
            g[(r["condition"], r.get("budget"), r["model"])].append(r)
    for k in sorted(g, key=lambda x: (x[0], x[1] or 0, x[2])):
        pa, ev, n, bits = paired(g[k])
        rr = sum(bool(x.get("used_reconstruction")) for x in g[k]) / len(g[k])
        print(f"   {k[0]:<3} B={str(k[1]):<3} {MODELS[k[2]]:<12} pair={pa:.2f} ev={ev:.2f} "
              f"recon={rr:.2f} n={n} {bits}")

rule("3. M4 COMPOUND (B=32)")
B4 = by_interface(rows, "compound")
for cond in ("A1", "A2", "A3"):
    for m in MODELS:
        if (cond, m) in B4:
            pa, ev, n, bits = B4[(cond, m)]
            g = [r for r in rows if r["mechanism"] == "compound" and r["condition"] == cond
                 and r["model"] == m]
            print(f"   {cond:<3} {MODELS[m]:<12} pair={pa:.2f} ev={ev:.2f} gap={pa - ev:.2f} "
                  f"calls={sum(x['tool_calls'] for x in g) / len(g):.1f} n={n} {bits}")

rule("4. EVIDENCE-VALIDITY GAP BY INTERFACE (M1-M3 pooled, per cell then averaged)")
cells = defaultdict(list)
for r in rows:
    if r["mechanism"] == "compound" or r["model"] not in MODELS:
        continue
    cells[(r["model"], r["condition"], r["mechanism"], r.get("budget"),
           r.get("sweep"))].append(r)
agg = defaultdict(list)
for k, v in cells.items():
    pa, ev, n, _ = paired(v)
    if n:
        agg[(k[0], k[1])].append((pa, ev))
for k in sorted(agg):
    vals = agg[k]
    mpa = sum(v[0] for v in vals) / len(vals)
    mev = sum(v[1] for v in vals) / len(vals)
    print(f"   {MODELS[k[0]]:<12} {k[1]:<3} cells={len(vals):<3} "
          f"pair={mpa:.2f} ev={mev:.2f} gap={mpa - mev:.2f}")

rule("4b. A3 EVIDENCE GAP PER MECHANISM (headline cells only, n=5)")
# A grand mean over all A3 cells lets the 20-cell ambiguity sweep dominate mini's average and is
# not comparable to Opus, which never ran it. The paper reports per mechanism instead.
HEAD = {"cascade": ("resolution=32", "resolution=128", "resolution=512"),
        "closure_wave": ("cross_mechanism=closure_wave",),
        "regime_switch": ("cross_mechanism=regime_switch",),
        "compound": ("compound=compound",)}
for mech, sweeps in HEAD.items():
    for m in MODELS:
        # baseline condition only -- the robustness cells reuse the `resolution` sweep name at
        # R=128, and pair_id does not encode scale, so they merge into this cell otherwise
        sel = [r for r in rows if r["mechanism"] == mech and r["condition"] == "A3"
               and r["model"] == m and r.get("sweep") in sweeps
               and r.get("confounder", "none") == "none" and r.get("scale", "dev") == "dev"]
        if not sel:
            continue
        pa, ev, n, bits = paired(sel)
        print(f"   {mech:<14}{MODELS[m]:<12} pair={pa:.2f} ev={ev:.2f} gap={pa - ev:.2f} n={n}")

# What the stored `cited_bad_support` flag measures on M1-M3: a semantic call with a missing or
# different bound is cited as support. Under the stated rule that alone does not make a run
# evidence-invalid; it does only when no cited support step is task-bound. Both counts are
# printed, with the rule named, so the co-citation behaviour stays visible. Rule-independent.
print("\n   off-constraint semantic call cited as support (A3 headline runs, M1-M3; both models):")
tot = [0, 0, 0]
for mech in ("cascade", "closure_wave", "regime_switch"):
    sel = [r for r in rows if r["mechanism"] == mech and r["condition"] == "A3"
           and r["model"] in MODELS and r.get("sweep") in HEAD[mech]
           and r.get("confounder", "none") == "none" and r.get("scale", "dev") == "dev"]
    off = [r for r in sel if r.get("cited_bad_support")]
    also = sum(ev_valid(r, "stated") for r in off)
    tot = [tot[0] + len(sel), tot[1] + len(off), tot[2] + also]
    print(f"   {mech:<14}runs={len(sel):<4} off-constraint cited={len(off):<3} "
          f"of which a task-bound call is also cited (stated-rule valid)={also}")
print(f"   {'M1-M3 total':<14}runs={tot[0]:<4} off-constraint cited={tot[1]:<3} "
      f"of which a task-bound call is also cited (stated-rule valid)={tot[2]}")

rule("4c. EXECUTED-SUPPORT ROBUSTNESS (ExecEVPairAcc, headline cells)")
# Final readiness review §5: recompute the evidence criterion over ALL executed steps, cited or
# not, to separate a task-faithful computation that was executed but not cited as support from
# the absence of one. Loads traces and recompiles pairs like scripts/rescore.py; rows whose world
# no longer matches are skipped.
import yaml                                                                     # noqa: E402
from evaluation.exec_support import exec_ev                                     # noqa: E402
from tasks.lookup import compiled_pair                                          # noqa: E402
from traces.schema import Trace                                                 # noqa: E402

_mcfg = yaml.safe_load((ROOT / "configs/mechanisms.yaml").read_text())
_xcfg = yaml.safe_load((ROOT / "configs/experiment_matrix.yaml").read_text())
_pair_cache: dict = {}


def _exec_paired(sel: list[dict]) -> tuple[float, float, int, int]:
    """(ExecEVPairAcc, EVPairAcc, n_pairs, n_skipped) over one cell's rows."""
    from collections import defaultdict as _dd
    by, skipped = _dd(dict), 0
    for r in sel:
        tp = ROOT / "runs" / r["_dir"] / "traces" / f"{r.get('run_id', '')}.json"
        if not tp.exists():
            skipped += 1
            continue
        trace = Trace.from_dict(json.loads(tp.read_text()))
        cp = compiled_pair(r, _mcfg, _xcfg, _pair_cache)
        rec = getattr(cp, r["polarity"])
        if rec.world_hash != trace.world_hash:
            skipped += 1
            continue
        tc = (cp.compound_tc or {}).get(r["polarity"])
        by[r["pair_id"]][r["polarity"]] = (r, trace, rec.task, tc)
    complete = [by[k] for k in sorted(by) if len(by[k]) == 2]
    if not complete:
        return 0.0, 0.0, 0, skipped
    xok = sum(exec_ev(*v["positive"][:3], compound_tc=v["positive"][3])
              and exec_ev(*v["negative"][:3], compound_tc=v["negative"][3]) for v in complete)
    evok = sum(ev_valid(v["positive"][0]) and ev_valid(v["negative"][0]) for v in complete)
    return xok / len(complete), evok / len(complete), len(complete), skipped


for mech, sweeps in HEAD.items():
    for cond in (("A2", "A3") if mech == "compound" else ("A3",)):
        for m in MODELS:
            sel = [r for r in rows if r["mechanism"] == mech and r["condition"] == cond
                   and r["model"] == m and r.get("sweep") in sweeps
                   and r.get("confounder", "none") == "none" and r.get("scale", "dev") == "dev"]
            if not sel:
                continue
            xev, ev, n, sk = _exec_paired(sel)
            print(f"   {mech:<14}{cond} {MODELS[m]:<12} execEV={xev:.2f} citedEV={ev:.2f} "
                  f"n={n}" + (f" SKIPPED={sk}" if sk else ""))

rule("5. OPEN-WEIGHT LADDER")
lad = defaultdict(list)
for r in rows:
    if r["model"] in LADDER:
        lad[(r["model"], r["condition"], r.get("sweep"))].append(r)
proto = defaultdict(int)
for r in rows:
    if r["model"] in LADDER:
        proto[r["model"]] += r.get("tool_protocol_errors", 0) or 0
for k in sorted(lad, key=lambda x: (LADDER[x[0]], x[1], str(x[2]))):
    pa, ev, n, bits = paired(lad[k])
    fab = sum(1 for r in lad[k] if r.get("failure_category") == "fabricated_evidence") / len(lad[k])
    print(f"   {LADDER[k[0]]:<13} {k[1]:<3} {str(k[2]):<28} pair={pa:.2f} fabricate={fab:.2f} n={n}")
print("\ntool-protocol errors by checkpoint:", {LADDER[k]: v for k, v in proto.items()})
print("open-weight episodes:", sum(1 for r in rows if r["model"] in LADDER))

rule("6. COST DECOMPOSITION")
bc = []
for f in (ROOT / "runs").glob("*/backend_cost.jsonl"):
    if "INVALID" in f.parent.name or f.parent.name in ("m3_mini", "m3_opus"):
        continue
    for line in f.read_text().splitlines():
        if line.strip():
            bc.append(json.loads(line))
g = defaultdict(list)
for r in bc:
    if r["model"] in MODELS:
        g[(r["model"], r["condition"])].append(r)
print(f"   {'model':<12}{'cond':<5}{'runs':>5}{'llm':>7}{'tools':>7}{'inTok':>9}{'$/run':>8}"
      f"{'events':>12}{'ev/LLM':>10}{'bkendms':>9}")
for k in sorted(g):
    v = g[k]
    mean = lambda key: sum(x.get(key, 0) or 0 for x in v) / len(v)   # noqa: E731
    llm = mean("llm_calls")
    print(f"   {MODELS[k[0]]:<12}{k[1]:<5}{len(v):>5}{llm:>7.1f}{mean('tool_calls'):>7.1f}"
          f"{mean('input_tokens'):>9.0f}{mean('cost_usd'):>8.3f}{mean('rows_scanned'):>12,.0f}"
          f"{(mean('rows_scanned') / llm if llm else 0):>10,.0f}{mean('backend_ms'):>9.0f}")
lat = ROOT / "runs/latency_c1/results.jsonl"
if lat.exists():
    lr = [json.loads(l) for l in lat.read_text().splitlines() if l.strip()]
    by = defaultdict(list)
    for r in lr:
        by[r["condition"]].append(r)
    print("\n   controlled latency, concurrency 1 (gpt-5-mini, M1 R=32):")
    for c in sorted(by):
        v = by[c]
        print(f"     {c}  {sum(x['latency_ms'] for x in v) / len(v) / 1000:>6.1f} s/episode  "
              f"n={len(v)}")

rule("7. A0 IDENTIFIABILITY CONTROL")
a0 = [r for r in rows if r["condition"] == "A0"]
g = defaultdict(list)
for r in a0:
    g[r["model"]].append(r)
for m, v in sorted(g.items()):
    pa, ev, n, _ = paired(v)
    print(f"   {MODELS.get(m, m):<12} pairAcc={pa:.2f} over n={n} pairs ({len(v)} runs)")

rule("8. ROBUSTNESS CELLS (M1, R=128, B=16)")
# The robustness table in the paper. Cells are split by (confounder, scale) BEFORE pairing,
# because the robustness sweeps reuse `resolution=128` pair_ids (see the guard comments above).
g = defaultdict(list)
for r in rows:
    name, _, val = (r.get("sweep") or "").partition("=")
    if (name in ("resolution", "resolution_fine") and val == "128"
            and r["condition"] in ("A1", "A3") and r["model"] in MODELS):
        cond = (r.get("confounder", "none"), r.get("scale", "dev"))
        g[(cond, r["condition"], r["model"])].append(r)
LABEL = {("none", "dev"): "baseline", ("c1_global_burst_high", "dev"): "+confounder",
         ("none", "test"): "+scale (8K)"}
for key in sorted(g, key=str):
    v = g[key]
    pa, ev, n, bits = paired(v)
    rr = sum(bool(r.get("used_reconstruction")) for r in v) / len(v)
    calls = sum(r.get("tool_calls", 0) or 0 for r in v) / len(v)
    (conf_scale, cond, m) = key
    print(f"   {LABEL.get(conf_scale, str(conf_scale)):<13}{cond:<4}{MODELS[m]:<12} "
          f"pair={pa:.2f} recon={rr:.2f} calls={calls:>4.1f} n={n} {bits}")
