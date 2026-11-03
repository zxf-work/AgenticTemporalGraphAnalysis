# Capability Is Not Validity — anonymous artifact

Source code, configurations, preregistration records, and the complete run corpus for the WSDM
2027 submission *Capability Is Not Validity: Evaluating Agentic Temporal Graph Analysis*.

**Research question.** Agentic analysis of temporal graphs is evaluated along two independent
axes: what a model–interface system is *capable* of concluding, and whether the computation cited
in support of a successful conclusion actually *establishes* the constrained proposition asked
(trace-grounded evidence validity).

**Method.** Matched counterfactual world pairs `(G+, G-)`. `G-` is produced from `G+` by permuting
the *assignment* of the existing timestamp multiset to events, subject to constraints that break
the target temporal mechanism. Node set, static edge multiset, event count, and global timestamp
multiset are identical **by construction**; only the temporal assignment map differs. An agent
that does not use the decisive temporal evidence cannot separate the two worlds.

## Layout

```
configs/           mechanisms.yaml (world generators), experiment_matrix.yaml (frozen sweeps),
                   models.yaml
worlds/            temporal-world construction, invariant checks, mechanisms/ (M1-M4), confounders/
tasks/             task schema, served task templates, deterministic pair compiler
tools/             the four interface surfaces exposed to agents (A0-A3)
agents/            model backends, agent loop, tool schemas
traces/            trace schema (every episode stores its full trajectory)
evaluation/        deterministic scoring: checker, SemValid / error taxonomy, ExecEVPairAcc
                   (exec_support.py), metrics, shared figure data (figdata.py)
scripts/           analysis, figures, tests, and run orchestration (see below)
runs/              the complete corpus: results.jsonl + stored traces per sweep
figures/           the paper's figures, regenerated from runs/ by scripts/figure*.py
PREREGISTRATION*.md  cell/model/analysis declarations frozen before each experiment,
                   including all protocol amendments and one disclosed execution incident
```

The preregistration files are verbatim frozen records; they occasionally cite internal working
documents (e.g. `PAPER_CLAIMS.md`) that are not part of this artifact.

`data/` (compiled worlds and task records) is generated, not committed; it is regenerated
deterministically from `configs/mechanisms.yaml`, so a fresh clone reproduces byte-identical
worlds and world hashes. `runs/` holds the scored results and stored traces that every figure and
number in the paper is read from.

## Reproducing the paper

All analysis runs offline from the committed corpus; no model credentials are needed.

```
python scripts/paper_numbers.py        # every number quoted in the paper
python scripts/figure1.py              # M4 accuracy vs evidence validity
python scripts/figure2.py              # M1 capability frontier + reconstruction
python scripts/figure_framework.py     # staged profile and the four mechanisms
python scripts/bootstrap_contrast.py   # the one reported confidence interval
python scripts/rescore.py --all        # rebuild scored rows from stored traces
python scripts/replay_backend.py --all # backend cost via replay, no LLM calls
python scripts/diag_m4_anchor.py       # per-run M4 stage-2 windows vs the oracle t_c
python scripts/diag_m4_payload.py      # M4 audit-boundary sensitivity: do cited payloads decide it?
python scripts/check_m3_threshold.py   # M3 ablation: are labels monotone in the returned score?
```

Requires Python 3.11 with `numpy`, `matplotlib`, and `pyyaml` (`pip install -r
requirements.txt`). Only `scripts/run_matrix.py` (running *new* episodes) needs model
credentials: `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` via environment, or a local OpenAI-compatible
endpoint for the open-weight ladder (`scripts/ladder_job.sbatch` shows the vLLM serving job; set
your own SLURM account/partition).

## Corpus conventions

- The reported corpus is every scored run whose world recompiles to its recorded hash, minus
  superseded sweeps. Supersessions are explicit, never deletions: directories named
  `*_INVALID_*` / `*_PRE_AMENDMENT` and the `SKIP_DIRS` list in `evaluation/figdata.py` record
  exactly which sweeps were replaced and why; the preregistration documents record the
  corresponding protocol amendments.
- Run identifiers are content-derived (task, condition, model, prompt hash, world hash), so a
  configuration change that should change behavior must change run ids —
  `scripts/preflight_new_worlds.py` enforces this before any re-run.
- Counting. `runs/` holds 1,402 scored rows for 1,382 executed episodes and 1,350 distinct run
  identities. Twenty rows are cached copies: the preregistered Opus 5 anchor episodes at R=512
  (`anchors/`) are re-served by content hash into `resolution_opus/`, where the resolution curve
  needs them (`cached: true`). Thirty-two identities were executed twice in different sweeps
  (the gpt-5-mini `ambiguity=0` cells and the `resolution=512` cells are the same worlds, and
  `latency_c1/` re-runs twelve R=32 episodes at concurrency 1); the hosted models sample
  non-deterministically, so a few of those re-executions reached different decisions.
  `evaluation/figdata.corpus_counts` reports all three numbers; `latency_c1/` is reported only in
  the latency measurement and is excluded from every grid cell (`CELL_EXCLUDE_DIRS`).
- M4 scoring. The reported M4 evidence criterion is the anchored, claim-sensitive stage-2 rule,
  which needs the oracle closure time t_c recomputed from the world. The runner passes it to the
  scorer, and `scripts/rescore.py --all` is idempotent on the committed rows (0 changed). The
  history of that rule is recorded in `PREREGISTRATION_C5.md` (decision record, 2026-11-03).
  `rescore.py` recompiles worlds at dev scale, so the 8K-node rows (`r3b_scale_*`) are skipped
  by it and stand as scored at run time; they contain no M4 cells.
- Sampling. `configs/models.yaml` requests temperature 0, but the hosted frontier APIs reject
  the parameter and the backend drops it after the first refusal (`agents/backends.py`), so
  Opus 5 and gpt-5-mini ran at provider-default sampling; the Qwen ladder ran at temperature 0
  through vLLM. Each world is run once per sweep directory. The serving-controls file the ladder
  job writes (`runs/ladder_<slot>_controls.json`: vLLM, torch, GPU and driver versions) was not
  committed and is not recoverable from this artifact.
- `scripts/test_*.py` and `scripts/selftest_schemas.py` are the invariant and schema test suite.

## Anonymity

This artifact is anonymized for double-blind review: commits, traces, and configurations carry no
author-identifying information, and site-specific compute settings are replaced by placeholders.
