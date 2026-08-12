"""Protocol sanity checks for a completed sweep — a gate, not an analysis.

Every check here asks "was this run *valid*?", never "were the results *good*?". They are written
before the sweep is read so the gate cannot be tuned to whatever came back, and they are the
precondition for spending frontier budget on the preregistered anchors: validating a broken run
with an expensive model wastes the budget twice.

The A0 check is the one that can fail for an interesting reason. A0's entire purpose is that its
observable surface is identical across a matched pair, so an A0 pair accuracy meaningfully above
the 25% random-guessing ceiling means information is leaking through an interface that is supposed
to be order-oblivious — a benchmark defect, not a model result.

Run:  python scripts/sanity_check_sweep.py runs/<sweep>
"""

from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUN = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "runs/ambiguity")
EXPECT_RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 480

results: list[tuple[bool, str, str]] = []


def check(name: str, cond: bool, detail: str = "", note: str = "") -> None:
    """`detail` explains a failure and is printed only on failure; `note` always prints.

    An earlier version printed `detail` unconditionally, so a failure message sat next to an [ok]
    and read as though the check had failed. Keeping the two apart makes the gate's output mean
    exactly one thing.
    """
    results.append((cond, name, detail))
    suffix = f"  — {note}" if note else ("" if cond else f"  — {detail}")
    print(f"[{'  ok  ' if cond else ' FAIL '}] {name}{suffix}")


rows = {}
for line in (RUN / "results.jsonl").read_text().splitlines():
    if line.strip():
        d = json.loads(line)
        rows[d.get("run_id", line)] = d
rows = list(rows.values())

# ------------------------------------------------------------------ 1. completeness
check(f"expected run count ({EXPECT_RUNS})", len(rows) == EXPECT_RUNS, f"got {len(rows)}")

statuses = Counter(r["status"] for r in rows)
n_ok = statuses.get("success", 0)
check("at least 95% of runs succeeded", n_ok >= 0.95 * len(rows),
      f"{n_ok}/{len(rows)}  {dict(statuses)}")

# ------------------------------------------------------------------ 2. world isolation
# The bug that invalidated the first sweep: worlds differing only in a generator parameter shared
# a filename, so every cell read the same data and the axis looked flat.
by_sweep_pairs = defaultdict(set)
for r in rows:
    by_sweep_pairs[r.get("sweep", "")].add(r["pair_id"])
all_pairs = set().union(*by_sweep_pairs.values()) if by_sweep_pairs else set()
check("each axis value uses its own pair ids",
      all(len(a & b) == 0 for i, a in enumerate(by_sweep_pairs.values())
          for b in list(by_sweep_pairs.values())[i + 1:]),
      f"{len(all_pairs)} distinct pair ids across {len(by_sweep_pairs)} axis values")
check("pair count matches axis values x seeds",
      len(all_pairs) == len(by_sweep_pairs) * 5,
      f"{len(all_pairs)} pairs, expected {len(by_sweep_pairs) * 5}")

# every generated pair must have passed the invariant checker at compile time
# data/invariants accumulates across every sweep ever compiled, so this sweep's pairs are a
# subset of it. Check per-pair coverage of THIS sweep, not directory size: requiring equality
# would fail whenever any other sweep had run, and would say nothing about the pairs at hand.
#
# `all([])` is True, so each check below must also require its evidence to be non-empty —
# a missing artifact directory once passed three checks vacuously.
inv_dir = ROOT / "data/invariants"
by_pair = {}
if inv_dir.exists():
    for f in inv_dir.glob("*.json"):
        d = json.loads(f.read_text())
        by_pair[d.get("pair_id", f.stem)] = d
mine = [by_pair[p] for p in all_pairs if p in by_pair]
missing = sorted(all_pairs - set(by_pair))

check("invariant record exists for every pair in this sweep", not missing,
      f"missing {len(missing)}: {missing[:3]}",
      note=f"{len(mine)}/{len(all_pairs)} pairs covered ({len(by_pair)} records on disk)")
check("every pair in this sweep passed invariants",
      bool(mine) and len(mine) == len(all_pairs) and all(i["ok"] for i in mine),
      f"{sum(i['ok'] for i in mine)}/{len(mine)} ok")
check("static view identical across every pair in this sweep",
      bool(mine) and all(i.get("same_static_view", False) for i in mine))
check("every negative world in this sweep globally clean",
      bool(mine) and all(i.get("global_oracle_check_negative") == "pass" for i in mine))

# ------------------------------------------------------------------ 3. pairing integrity
polarity = defaultdict(set)
for r in rows:
    polarity[(r["pair_id"], r["condition"], r.get("budget"), r["model_slot"])].add(r["polarity"])
incomplete = [k for k, v in polarity.items() if v != {"positive", "negative"}]
check("every cell has both polarities", not incomplete, f"{len(incomplete)} incomplete")

# ------------------------------------------------------------------ 4. identifiability control
def pair_acc(subset) -> tuple[float, int]:
    d = defaultdict(dict)
    for r in subset:
        d[r["pair_id"]][r["polarity"]] = r["decision"]
    complete = [v for v in d.values() if len(v) == 2]
    if not complete:
        return 0.0, 0
    ok = sum(1 for v in complete
             if v["positive"] == "present" and v["negative"] == "absent")
    return ok / len(complete), len(complete)

# A0 is required only when the sweep declared it. Reading that from the manifest keeps a
# deliberate omission ("this sweep tests A1 vs A3") distinct from an accidental one ("the control
# silently failed to run") — a check that quietly skips itself is worse than no check.
manifest_conditions = []
mf = RUN / "manifest.json"
if mf.exists():
    manifest_conditions = json.loads(mf.read_text()).get("conditions", [])

a0 = [r for r in rows if r["condition"] == "A0"]
if a0:
    acc, n = pair_acc(a0)
    # 0.25 is the ceiling for independent guessing; allow slack for small n
    check("A0 sits at or below the identifiability ceiling", acc <= 0.35,
          f"A0 pair accuracy {acc:.2f} over {n} pairs (guessing ceiling 0.25)")
elif "A0" in manifest_conditions:
    check("A0 ran as declared", False, "manifest declares A0 but no A0 runs are present")
else:
    print("[  n/a ] A0 identifiability control — not declared for this sweep")

# ------------------------------------------------------------------ 5. instrumentation
# Same rule as the A0 control: required only when the sweep declared the condition. A sweep that
# deliberately tests one interface must not fail for lacking another.
n_a3_temporal = sum(1 for r in rows if r["condition"] == "A3" and r["n_temporal_calls"] > 0)
if any(r["condition"] == "A3" for r in rows):
    check("A3 actually used temporal operators", n_a3_temporal > 0,
          "A3 made no temporal calls at all",
          note=f"{n_a3_temporal} A3 runs used at least one temporal operator")
elif "A3" in manifest_conditions:
    check("A3 ran as declared", False, "manifest declares A3 but no A3 runs are present")
else:
    print("[  n/a ] A3 temporal-operator use — condition not declared for this sweep")
check("cost recorded per run", all("cost_usd" in r for r in rows))
spend = sum(r.get("cost_usd", 0) for r in rows)
models = Counter(r["model"] for r in rows)
print(f"\n       spend ${spend:.2f} across {dict(models)}")

# malformed traces would have broken json.loads above; check trace files parse
bad_traces = 0
for p in (RUN / "traces").glob("*.json"):
    try:
        json.loads(p.read_text())
    except Exception:
        bad_traces += 1
check("all trace files parse", bad_traces == 0, f"{bad_traces} corrupt")

n_fail = sum(1 for ok, _, _ in results if not ok)
print(f"\n{'=' * 70}\n{len(results) - n_fail}/{len(results)} protocol checks passed")
if n_fail:
    print("BLOCKED — do not spend frontier budget until these are resolved:")
    for ok, name, detail in results:
        if not ok:
            print(f"  - {name}: {detail}")
print(f"\nGATE: {'PASS — anchors may run' if not n_fail else 'FAIL — anchors blocked'}")
sys.exit(1 if n_fail else 0)
