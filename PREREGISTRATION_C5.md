# Preregistration — C5 compound investigation (M4)

**Committed before any M4 episode is run.**

## The objection this answers

*The framework only works because the benchmark is isolated synthetic predicates.* Gate G6 treats
one compound workload as the main protection against that reading.

## What M4 is

A two-stage investigation with a **dependency between the stages**:

    ClosureWave(C, W, delta) completes at t_c
    Cascade(s -> t, max_wait) begins within [t_c, t_c + admissible]

The counterfactual is built on the dependency and nothing else. Verified on the dev seeds before
committing this document:

| | closure fraction | t_c | cascade reachable at all | cascade inside admissible |
|---|---|---|---|---|
| G+ | 0.91–1.00 | e.g. 24 382 | yes | **yes** |
| G− | 0.91–1.00 | identical | yes | **no** |

Both worlds contain **both stages, individually satisfied**, with identical timestamp multisets,
identical static edge multisets and identical event counts. So the conjunctive strategy — "is
there a wave? is there a cascade? then yes" — answers *present* to both worlds and scores pair
accuracy **0.00**. Only an agent that computes t_c and then constrains the second stage to the
interval following it can separate the pair. That is the competence being measured.

`t_c` is **not** given to the agent. The delta, max_wait and admissible bounds *are* stated, as
they are in M1–M3: the dependency is the object of study, not a riddle about our intent.

## Cells

| mechanism | conditions | budget | seeds | models |
|---|---|---|---|---|
| compound | A1, A2, A3 | 32 | 5 | gpt-5-mini, Opus 5 |

60 episodes. A single generous budget (B=32) on purpose: a two-stage investigation legitimately
needs more calls, and the failure under test is compositional, so it must not be confounded with
running out of budget. No parameter sweep — the milestone forbids one here.

Budget cap for this milestone: **$25**. mini runs first; Opus runs under a hard
ceiling with reduced concurrency, because the ceiling stops *new* episodes but lets in-flight ones
finish and has previously overshot by ~25% at concurrency 12.

## Measurements

PairAcc, **EVPairAcc**, per-seed outcomes, tool calls, tokens, cost, failure mix, and the
diagnostic that matters here: **how often a run reaches the right answer without ever binding the
second stage to t_c**. EVPairAcc is expected to expose exactly that — a semantically invalid plan
behind a correct final answer.

## Predictions (recorded before the run)

1. PairAcc falls for every interface relative to the single-stage mechanisms; the compound task is
   strictly harder.
2. The gap between PairAcc and EVPairAcc is **larger** on M4 than on M1–M3, because a correct
   answer can be reached by a plan that never binds the stages.
3. A3 keeps its interaction-cost advantage but does not rescue compositional correctness: having
   `temporal_closure` and `temporal_reachability` as operators does not tell an agent to feed the
   first's output into the second's window.
4. A2 does comparatively better here than on the single-stage mechanisms, because the dependency
   is naturally expressed in code.

## Outcomes, all publishable

* **All interfaces collapse** — the framework's diagnosis holds but the workload is out of reach;
  report as a bound on current agents, not a benchmark defect.
* **A2 > A3 > A1** — programmability wins when stages must be composed.
* **A3 holds up** — semantic operators compose better than expected; weakens prediction 3 and is
  pre-committed to.

## Rules

* No prompt tuning, no parameter retuning after seeing results. The templates and the config are
  frozen with this document.
* Results update the *scope* sentences of `PAPER_CLAIMS.md`; the three findings are frozen.
* Background topology is synthetic, as for M1–M3. The milestone prefers real topology "if
  practical"; it was not attempted within this deadline and is reported as a limitation rather
  than quietly omitted.

## Decision record, 2026-11-03 (camera-ready; appended, nothing above edited)

**What this document fixed before the runs, and what it did not.** The cells, budget, models,
the metrics (PairAcc, EVPairAcc) and the diagnostic "how often a run reaches the right answer
without ever binding the second stage to t_c" were declared above before any M4 episode. The
*operational rule* for "binding the second stage to t_c" was not declared here. Two rules have
existed since:

* At first scoring and at submission, the stage-2 test was a **width-only proxy**
  (`error_taxonomy._compound_stage2_bound`): a cited reachability window had to be a strict
  sub-interval of the observation window no wider than 1.5 x the admissible lag. Under it the
  M4 A3 cells scored EVPairAcc 0.60 (Opus 5) and 0.00 (gpt-5-mini).
* During the pre-submission language review (2026-08-13, after the traces had been read) the
  rule was replaced by the **anchored, claim-sensitive test**
  (`error_taxonomy._compound_stage2_anchored`): the cited window must lie within the admissible
  interval [t_c, t_c + admissible] padded by a quarter of the lag on each side, and an absence
  claim must additionally cover the whole interval; t_c is recomputed from the world at scoring
  time. Rescoring changed six Opus 5 A3 rows and moved Opus 5 to EVPairAcc 0.20; gpt-5-mini,
  A1 and A2 were unchanged. The reported M4 numbers use this rule.

The second rule is the stricter and more faithful reading of the diagnostic declared above, but
it was fixed after the data were seen, and this record exists so that the paper can say so.

**Boundary sensitivity, recorded for the camera-ready.** The decisive M4 inference, comparing
returned arrival times with [t_c, t_c + admissible], is itself deterministic. Recomputed over the
cited reachability and paths payloads of all twenty A3 episodes (`scripts/diag_m4_payload.py`),
that comparison yields the gold label in 19 of 20 episodes (gpt-5-mini 4 of 5 pairs under the
strict reading, 5 of 5 if a bounded path enumeration that returned fewer than `max_paths` routes
counts as exhaustive; Opus 5 5 of 5). This is reported as a sensitivity of the audit boundary,
not as a replacement for the anchored criterion, which deliberately does not credit a
composition performed in prose over returned values.

**Provenance caveat.** This artifact repository was assembled on 2026-08-12 from a development
repository whose history is not included; the preregistration and the M4 traces were imported
minutes apart, so the git history here cannot by itself show that this document preceded the
episodes. The development repository's commits are referenced in the run manifests
(`synced_commit`).
