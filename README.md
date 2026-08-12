# temporal-agent-probes

Counterfactual temporal-graph probes for evaluating LLM agents (WSDM 2027, paper P0a).

**Research question.** Can an LLM agent distinguish a genuine temporal mechanism in an evolving
graph from static-structure, recency, popularity, and activity-volume shortcuts?

**Method.** Matched counterfactual world pairs `(G+, G-)`. `G-` is produced from `G+` by permuting
the *assignment* of the existing timestamp multiset to events, subject to constraints that break the
target mechanism. Node set, static edge multiset, event count, and global timestamp multiset are
therefore identical **by construction**; only the temporal assignment map differs. An agent that
does not use the decisive temporal evidence cannot separate the two worlds.

## Execution environment

All computation runs on the remote host, never locally.

| | |
|---|---|
| Compute host | `${ATGP_HOST:?set to your compute host}` (user `$USER`), 40 cores, 93 GB RAM |
| Env | conda env `atgp`, Python 3.11.15 |
| Remote repo | `~/temporal-agent-probes` |
| Sync | `bash scripts/sync.sh` (rsync push from this local checkout) |

## Layout

```
configs/     models.yaml, agents.yaml, mechanisms.yaml, experiment_matrix.yaml
worlds/      base.py, pair.py, invariants.py, mechanisms/, confounders/
tasks/       schema.py, templates.py, compiler.py, frozen_seeds.txt
tools/       static_surface.py (A1), temporal_surface.py (A3/A4), oracles
agents/      harness.py, generic_tools.py, code_agent.py, temporal_tools.py, contrastive_evidence.py
traces/      schema.py
evaluation/  claim_schema.py, checker.py, metrics.py, bootstrap.py, error_taxonomy.py
scripts/     generate_dev.py, generate_test.py, run_smoke.py, run_matrix.py, score.py, make_figures.py,
             leakage_audit.py, sync.sh
```

## Governing documents

Execution follows `../wsdm2027_p0a_execution_taskboard.md` (authoritative; overrides the plan on
conflict) and `../wsdm2027_temporal_graph_agent_paper1_plan.md` (the plan).

## Originality boundaries (plan §3 — non-negotiable)

**This work may claim:** a counterfactual temporal-pair evaluation methodology; three temporal
mechanism probe families; a controlled study of temporal shortcut reliance; a small
evidence-grounding protocol; empirical findings about graph agents under temporal confounders.

**This work may NOT claim:** a general agent-native graph algebra; a complete Claim IR; a
comprehensive temporal-graph benchmark; adaptive exploration or planning; autonomous hypothesis
generation; FDR or selective inference; standing-goal monitoring; agent memory or runtime
optimization. Those are reserved for later papers (P1, P0/ATG-Bench, P2, P3, P4, P5).

## Hard rules

1. Scope firewall — no new mechanisms, models, memory, planning, GraphRAG, or statistical-discovery
   features before submission (plan §23).
2. No test-set tuning after the freeze commit (Aug 18). Dev seeds `0..4` only, before then.
3. Every run logs a full trajectory (`traces/schema.py`) from the first smoke test onward.
4. Determinism — every generator seeded; every run manifest records world hash, prompt hash, model
   version, and code commit; all reported numbers regenerate from one results artifact.
5. Pair accuracy above 95% for a non-oracle agent is a leakage alarm, not a success. Audit first.
