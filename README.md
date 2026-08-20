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
                   models.yaml, agents.yaml
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
- `scripts/test_*.py` and `scripts/selftest_schemas.py` are the invariant and schema test suite.

## Anonymity

This artifact is anonymized for double-blind review: commits, traces, and configurations carry no
author-identifying information, and site-specific compute settings are replaced by placeholders.
