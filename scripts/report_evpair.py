"""Milestone C2 — Evidence-Valid Pair Accuracy, computed over completed runs.

    python scripts/report_evpair.py runs/resolution_opus runs/xmech_opus ...
    python scripts/report_evpair.py --all
    python scripts/report_evpair.py --all --ev-rule submitted

Reports evidence-valid pair accuracy beside plain pair accuracy so the gap is visible:

    EV_i        = [answer correct] AND [no cited step is nonexistent]
                  AND [at least one cited support step satisfies the task-constraint test]
    EVPairAcc   = (1/N) sum_i EV_i^+ EV_i^-

Every conjunct is deterministic. No LLM judge is used anywhere in this file, per the milestone's
requirement that deterministic parameter checks are the primary validity oracle.

What makes a run evidence-INVALID under this rule (the one the paper states):

  * cited a step that does not exist -- a nonexistent-step citation. This is the dominant failure
    of the smaller open-weight subjects, which emit a well-formed claim citing tool calls they
    never made.
  * cited no support step at all, whatever the decision.
  * no cited support step is bound to the task's constraint -- e.g. only unconstrained
    reachability is offered as support for a delta-constrained question.

What does NOT make a run invalid: citing an exploratory or off-bound call BESIDE a task-bound
one, narrowing a window, or probing an intermediate node. `diffQ` still reports how often an
off-bound call is cited as support, and `decomp` how often a call is a decomposition probe.

`--ev-rule submitted` scores with the rule the reviewed submission used
(`check.evidence_grounded and not cited_bad_support`; see `evaluation/ev_rule.py`), which fails an
A3 run on M1-M3 as soon as any off-bound semantic call is cited as support. The stated rule is
derived from stored traces and recompiled worlds, so under it rows whose world no longer
recompiles (the superseded sweeps) are left out and counted.

Note EVPairAcc <= PairAcc by construction: it adds conjuncts to a correct answer.
"""

from __future__ import annotations

import json
import pathlib
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import figdata                                                  # noqa: E402
from evaluation.ev_rule import STATED, ev_rule_from_argv                        # noqa: E402

EV_RULE = ev_rule_from_argv(sys.argv)
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if "--all" in sys.argv or not args:
    dirs = sorted(d for d in (ROOT / "runs").glob("*") if (d / "results.jsonl").exists())
else:
    dirs = [pathlib.Path(a) for a in args]

rows: list[dict] = []
for d in dirs:
    seen = {}
    for line in (d / "results.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            r["_dir"] = d.name
            seen[r.get("run_id", line)] = r
    rows.extend(seen.values())

if not rows:
    sys.exit("no results found")


def ev_valid(r: dict) -> bool:
    """EV_i for one run under the rule in force. Deterministic; mirrors RunRow.ev_valid."""
    return figdata.ev_valid(r, EV_RULE)


if EV_RULE == STATED:
    # The stated rule needs each run's trace and recompiled task record. Superseded sweeps no
    # longer recompile to their recorded worlds; they cannot be scored and are left out.
    kept, unscorable = [], 0
    for r in rows:
        try:
            figdata.trace_and_record(r)
            kept.append(r)
        except (LookupError, KeyError, OSError, ValueError):
            unscorable += 1
    rows = kept
    if unscorable:
        print(f"{unscorable} rows left out: their world no longer recompiles or their trace is "
              f"missing (superseded sweeps); `--ev-rule submitted` scores them from stored rows\n")


# The deterministic reference algorithm sets only {"decision": ...} and never populates an
# evidence list, so every EV conjunct about citations fails for structural reasons that say
# nothing about analysis. Reporting it as EVPairAcc 0.00 beside the models would be a category
# error, so it is excluded here and its interaction cost is reported elsewhere.
REFERENCE_MODELS = {"script-a1-bisect"}
skipped_ref = sum(1 for r in rows if r.get("model") in REFERENCE_MODELS)
rows = [r for r in rows if r.get("model") not in REFERENCE_MODELS]

groups: dict[tuple, list[dict]] = defaultdict(list)
for r in rows:
    groups[(r.get("mechanism", "?"), r.get("model", "?"), r.get("condition", "?"),
            r.get("budget"), r.get("sweep", ""))].append(r)

print(f"evidence-validity rule: {EV_RULE}")
print(f"{len(rows)} runs across {len(dirs)} run dirs"
      + (f"  ({skipped_ref} reference-algorithm runs excluded: no evidence list by design)"
         if skipped_ref else "") + "\n")
hdr = (f"{'mechanism':<14}{'model':<26}{'cond':<5}{'B':>4}{'N':>4}"
       f"{'PairAcc':>9}{'EVPair':>8}{'gap':>7}{'constrErr':>10}{'diffQ':>7}{'decomp':>8}"
       f"{'fabric':>8}")
print(hdr)
print("-" * len(hdr))

totals: dict[tuple, list] = defaultdict(list)
for k in sorted(groups, key=lambda x: (str(x[0]), str(x[1]), str(x[2]), x[3] or 0)):
    mech, model, cond, budget, sweep = k
    g = groups[k]
    by_pair: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in g:
        by_pair[r["pair_id"]][r["polarity"]] = r
    complete = [v for v in by_pair.values() if len(v) == 2]
    if not complete:
        continue
    n = len(complete)
    pair_ok = sum(1 for v in complete
                  if v["positive"]["decision"] == "present"
                  and v["negative"]["decision"] == "absent")
    ev_ok = sum(1 for v in complete
                if v["positive"]["decision"] == "present"
                and v["negative"]["decision"] == "absent"
                and ev_valid(v["positive"]) and ev_valid(v["negative"]))

    with_temporal = [r for r in g if r.get("n_temporal_calls", 0) > 0]
    constr = (sum(r.get("param_error_rate", 0) for r in with_temporal) / len(with_temporal)
              if with_temporal else 0.0)
    diffq = (sum(bool(r.get("cited_bad_support")) for r in with_temporal) / len(with_temporal)
             if with_temporal else 0.0)
    decomp = (sum(r.get("decomposition_rate", 0) for r in with_temporal) / len(with_temporal)
              if with_temporal else 0.0)
    # Fabrication means a claim WAS produced and cited steps that do not exist. `citations_exist`
    # alone cannot say that: the checker's no-claim early-return sets every flag False, so a run
    # that simply ran out of budget would otherwise be reported as fabricating evidence -- which
    # is what it did to our own deterministic reference algorithm on the first pass.
    produced = [r for r in g if r.get("decision") not in (None, "none", "")]
    fabric = (sum(1 for r in produced if not (r.get("check") or {}).get("citations_exist", True))
              / len(produced)) if produced else 0.0

    pa, eva = pair_ok / n, ev_ok / n
    print(f"{mech:<14}{model:<26}{cond:<5}{budget or 0:>4}{n:>4}"
          f"{pa:>9.2f}{eva:>8.2f}{pa - eva:>7.2f}{constr:>10.2f}{diffq:>7.2f}{decomp:>8.2f}"
          f"{fabric:>8.2f}")
    totals[(model, cond)].append((pa, eva))

print("\nPairAcc vs EVPairAcc by model x interface (mean over cells):")
print(f"  {'model':<26}{'cond':<5}{'cells':>6}{'PairAcc':>9}{'EVPair':>8}{'gap':>7}")
for (model, cond) in sorted(totals):
    vals = totals[(model, cond)]
    mpa = sum(v[0] for v in vals) / len(vals)
    mev = sum(v[1] for v in vals) / len(vals)
    print(f"  {model:<26}{cond:<5}{len(vals):>6}{mpa:>9.2f}{mev:>8.2f}{mpa - mev:>7.2f}")
