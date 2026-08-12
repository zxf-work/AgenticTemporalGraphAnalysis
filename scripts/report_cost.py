"""Milestone C6 — controlled cost decomposition.

    python scripts/report_cost.py --all

Reports the three cost families separately, because collapsing them is how an interface comparison
goes wrong:

  **Agentic interaction cost** — LLM rounds, tool calls, primitive range probes, tokens, dollars.
  This is what the agent spends talking.

  **Backend cost** — events scanned and time inside the operator implementations, measured by
  serial replay (`scripts/replay_backend.py`), never from the concurrent sweep.

  **End-to-end latency** — reported only from runs recorded at controlled low concurrency, and
  labelled as such. Sweep wall-clock at concurrency 8-12 is contaminated by queueing and is
  excluded from the efficiency argument entirely, per C6's explicit instruction.

The number the paper needs from this is the events scanned *per LLM call*: an interface can look
efficient because the agent talks less while each utterance triggers far more work underneath.
"""

from __future__ import annotations

import json
import pathlib
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]

args = [a for a in sys.argv[1:] if not a.startswith("--")]
if "--all" in sys.argv or not args:
    dirs = sorted(d for d in (ROOT / "runs").glob("*") if (d / "backend_cost.jsonl").exists())
else:
    dirs = [pathlib.Path(a) for a in args]

rows: list[dict] = []
for d in dirs:
    f = d / "backend_cost.jsonl"
    if not f.exists():
        continue
    for line in f.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            r["_dir"] = d.name
            rows.append(r)

if not rows:
    sys.exit("no backend_cost.jsonl found -- run scripts/replay_backend.py first")

# range probes: A1's characteristic primitive, counted from the scored rows where available
probe: dict[str, float] = {}
for d in dirs:
    rf = d / "results.jsonl"
    if not rf.exists():
        continue
    for line in rf.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            probe[r.get("run_id", "")] = r.get("range_calls", 0)

groups: dict[tuple, list[dict]] = defaultdict(list)
for r in rows:
    groups[(r["model"], r["condition"])].append(r)


def mean(g, k):
    return sum(x.get(k, 0) or 0 for x in g) / len(g)


print(f"{len(rows)} runs\n")
print("AGENTIC INTERACTION COST".ljust(64) + "| BACKEND COST")
hdr = (f"{'model':<26}{'cond':<5}{'runs':>5}{'llm':>6}{'tools':>7}{'probes':>7}"
       f"{'inTok':>9}{'outTok':>8}{'$/run':>8}  | {'events':>12}{'ms':>9}{'ev/LLM':>10}")
print(hdr)
print("-" * len(hdr))
for k in sorted(groups, key=lambda x: (x[0], x[1])):
    g = groups[k]
    ev, llm = mean(g, "rows_scanned"), mean(g, "llm_calls")
    pr = sum(probe.get(x["run_id"], 0) for x in g) / len(g)
    print(f"{k[0]:<26}{k[1]:<5}{len(g):>5}{llm:>6.1f}{mean(g,'tool_calls'):>7.1f}{pr:>7.1f}"
          f"{mean(g,'input_tokens'):>9.0f}{mean(g,'output_tokens'):>8.0f}"
          f"{mean(g,'cost_usd'):>8.3f}  | {ev:>12,.0f}{mean(g,'backend_ms'):>9.0f}"
          f"{(ev / llm if llm else 0):>10,.0f}")

print("\nThe last column is the one that matters for an efficiency claim: events the data system\n"
      "examined per LLM round. A low tool-call count is not efficiency if each call scans the\n"
      "whole log.")

print("\nEnd-to-end latency: NOT reported here. Every sweep above ran at concurrency 8-12, so its\n"
      "wall-clock includes queueing and provider contention. C6 requires latency from a small\n"
      "representative set at controlled low concurrency; see runs/latency_c1 if present.")

lat = ROOT / "runs/latency_c1/results.jsonl"
if lat.exists():
    lrows = [json.loads(l) for l in lat.read_text().splitlines() if l.strip()]
    by: dict[tuple, list[dict]] = defaultdict(list)
    for r in lrows:
        by[(r["model"], r["condition"])].append(r)
    print(f"\nMeasured at concurrency 1 ({len(lrows)} runs):")
    print(f"  {'model':<26}{'cond':<5}{'runs':>5}{'s/episode':>11}{'s/LLM call':>12}")
    for k in sorted(by, key=str):
        g = by[k]
        s = sum(r["latency_ms"] for r in g) / len(g) / 1000
        c = sum(r.get("llm_calls", 0) for r in g) / len(g)
        print(f"  {k[0]:<26}{k[1]:<5}{len(g):>5}{s:>11.1f}{(s / c if c else 0):>12.1f}")
