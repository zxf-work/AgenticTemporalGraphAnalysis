# Preregistration — robustness cells (confounder and scale)

**Committed before either cell is run.** These are the last two experiments; after them the
experimental programme is closed.

## Why these two, and why only these two

Both answer predictable reviewer questions rather than testing a new claim. Neither is expected to
change the paper's findings, and both are reported whichever way they fall.

### R3a — confounder stress

The matched-pair construction already neutralises volume, popularity and static-structure
shortcuts **by construction**: both worlds carry identical node sets, static edge multisets, event
counts and timestamp multisets, so any such heuristic scores exactly 0.00. The confounder is a
stress test of that argument, not the primary control.

**Note on provenance.** `worlds/confounders/` was an empty directory: the overlays were declared in
the config and referenced by the compiler but never implemented, so no confounder cell could ever
have run. They were written for this preregistration and verified to preserve every pair invariant
(5/5 seeds feasible at both severities; static view identical; negative world globally clean).

Cell: **M1 cascade, R=128, B=16, A1 and A3, both principal models, 5 seeds,
`c1_global_burst_high`** — a broad activity spike sized at a multiple of the world's own
background rate, applied identically to both worlds. R=128 is chosen because Opus+A1 scores 1.00
there, so a degradation would be visible; a cell already at 0.00 could not show one.

### R3b — scale robustness

Every result so far is at development scale (2,000 nodes, ~16k events). This is **not** a
scalability study and must not be reported as one: 2K→8K is a robustness check that the observed
interface behaviour is not an artifact of a small graph.

Cell: **M1 cascade, R=128, B=16, A1 and A3, both principal models, 5 seeds, `scale=test`**
(8,000 nodes, ~64k events).

## Predictions

1. **R3a**: pair accuracy is unchanged within seed noise for both interfaces. The overlay adds
   events to both worlds identically, so it cannot carry pair-discriminating information.
2. **R3a, secondary**: interaction cost rises (more events to scan and summarise) without
   accuracy changing.
3. **R3b**: A3 stays at 1.00. A1 may *fall* for Opus, because a 4× denser graph makes each
   window query less selective and reconstruction more expensive at a fixed budget of 16.

Prediction 3 is the interesting one: if A1 falls at test scale, the frontier is partly a function
of graph size as well as temporal span, which **bounds** the M1 result rather than confirming it.
That would be reported as a limitation on the R curve's generality.

## Rules

* No tuning after seeing the outcome. If R3b degrades A1, that is the finding.
* mini runs first; Opus follows under a hard budget ceiling.
* Results update the *scope* sentences of `PAPER_CLAIMS.md` and the paper's Threats section.
