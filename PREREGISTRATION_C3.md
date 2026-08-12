# Preregistration — C3 A2 programmable baseline

**Committed before any A2 episode is run.**

## The objection this answers

*A3 only wins because A1 is an artificially weak alternative.* Gate G3 treats a programmable
interface as a necessary baseline given the current literature, so without A2 the interface
comparison is not credible regardless of how the numbers fall.

## What A2 is

Raw `(u, v, t)` events in a sandboxed Python interpreter, and no temporal operators. A2 therefore
gets **strictly more information than A1** — timestamps are handed over directly — and full
generality. It has **no reconstruction frontier by construction**. That is deliberate: the
question is not whether A2 can see time, but whether synthesising a correct time-respecting
algorithm under a call budget is cheaper or more reliable than specifying a semantic operator.

One tool (`run_python`), because A2's burden is to synthesise an algorithm; a menu of helpers
would blur it back toward A1/A3. State is not preserved between calls, so each program is
self-contained — recorded here because it is a real constraint on what A2 can do per call, and it
is stated plainly in the tool description and the prompt.

## Cells (verbatim from milestone C3)

| sweep | mechanism | axis | values | B |
|---|---|---|---|---|
| `a2_m1` | M1 cascade | R | 32, 256, 512 | 16 |
| `a2_m2` | M2 closure wave | B | — | 8, 32 |

5 seeds × 2 polarities. 50 episodes per model. Held at the same values as the existing A1/A3
columns, so A2 drops in beside them.

**Models.** gpt-5-mini first, both to shake down the interface and to measure real per-episode
cost before committing frontier budget. Opus 5 second, subject to the budget available at that
point. If only one model is affordable, A2 is reported for that model and the omission is
stated, not hidden.

## Measurements

PairAcc, EVPairAcc, executions per episode, **code-repair rate** (share of `run_python` calls
raising or timing out), tokens, wall-clock, and the failure mix.

## Predictions (recorded before the run)

1. A2 removes A1's resolution frontier for the strong model: A2 PairAcc is roughly flat in R,
   unlike A1's decline, because timestamps need no reconstruction.
2. A2 still requires frontier-model reasoning — the weak model does materially worse than the
   strong one under A2, more like A1 than A3.
3. A3 remains more interaction-efficient (fewer calls, fewer tokens) than A2.
4. A2 shows a failure mode absent from A1 and A3: wrong or mis-bound algorithms that execute
   cleanly and produce a confident wrong answer.

## Outcomes, all publishable

* **A2 ≈ A3** — abstraction is not what matters; interaction cost and evidence validity are. This
  weakens the paper's headline and is pre-committed to.
* **A2 between A1 and A3** — the expected result: programmability removes the acquisition frontier
  but adds a synthesis burden.
* **A2 < A1** — programmability is not free; report why from the traces.

## Rules

* Representative points only. No full A2 grid (milestone stop rule).
* No prompt tuning after seeing A2 results. The A2 paragraph is frozen with this document.
* Results update the *scope* sentences of `PAPER_CLAIMS.md`; the three findings are frozen.
