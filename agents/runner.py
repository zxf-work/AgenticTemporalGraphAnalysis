"""Concurrent, resumable experiment runner (WP8).

Structure follows from where the time goes. World generation is cheap (~0.2 s) but writes files,
so it runs **serially** up front; agent episodes are 40–180 s of almost pure network wait, so they
run **concurrently**. Splitting the two removes the only real race (two threads compiling the same
pair) without needing locks around the generator.

Guarantees
----------
* **Resumable.** Every run is keyed by ``run_id = f(task, condition, model, prompt_hash,
  world_hash)``. A completed run on disk is reused, so an interrupted matrix restarts where it
  stopped and a re-run with one new cell costs only that cell.
* **Isolated failures.** One episode raising never aborts the matrix; it is recorded with its
  status and the sweep continues.
* **Reproducible.** A manifest records the config, code commit, model versions and cell list;
  results land in a single JSONL that every table and figure is regenerated from.

Note on determinism: the frontier models in use reject a temperature setting, so runs are not
bitwise reproducible. Reproducibility rests on the released traces, worlds and prompts, and the
paper says so rather than implying replayability it does not have.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agents.harness import run_episode, system_prompt
from common import digest_text, stable_hash
from evaluation.cost import BudgetExceeded, BudgetGuard
from evaluation.metrics import RunRow, summarize
from evaluation.score_run import score
from tasks.compiler import MECH_KEY, compile_pair, load_world
from tasks.schema import TaskRecord
from traces.schema import Trace, make_run_id

ROOT = pathlib.Path(__file__).resolve().parents[1]


# ======================================================================================
# cells
# ======================================================================================

@dataclass(frozen=True)
class Cell:
    """One (world, interface, model, budget) combination. Expands to two runs, +/- polarity."""
    mechanism: str
    seed: int
    confounder: str
    condition: str
    model_slot: str
    budget: int
    paraphrase: int = 1
    scale: str = "dev"
    sweep: str = ""            # label for the grid axis this cell belongs to
    #: generator parameter overrides for this cell, e.g. (("hops", 7),). A tuple rather than a
    #: dict so the cell stays hashable and can key the world cache.
    overrides: tuple[tuple[str, Any], ...] = ()

    @property
    def world_key(self) -> str:
        """Cells sharing this key share a compiled world pair."""
        ov = ",".join(f"{k}={v}" for k, v in sorted(self.overrides))
        return (f"{self.mechanism}|{self.seed}|{self.confounder}|{self.paraphrase}|"
                f"{self.scale}|{ov}")

    def key(self) -> str:
        return (f"{self.world_key}|{self.condition}|{self.model_slot}|B{self.budget}")


def expand_grid(*, mechanisms: Iterable[str], seeds: Iterable[int],
                confounders: Iterable[str], conditions: Iterable[str],
                model_slots: Iterable[str], budgets: Iterable[int],
                paraphrases: Iterable[int] = (1,), scale: str = "dev",
                sweep: str = "") -> list[Cell]:
    return [Cell(m, s, c, cond, ms, b, p, scale, sweep)
            for m in mechanisms for s in seeds for c in confounders
            for cond in conditions for ms in model_slots for b in budgets for p in paraphrases]


# ======================================================================================
# runner
# ======================================================================================

@dataclass
class RunnerStats:
    total: int = 0
    done: int = 0
    cached: int = 0
    failed: int = 0
    started: float = field(default_factory=time.time)

    def line(self) -> str:
        el = time.time() - self.started
        rate = self.done / el if el > 0 else 0
        left = (self.total - self.done) / rate if rate > 0 else 0
        return (f"{self.done}/{self.total} done ({self.cached} cached, {self.failed} failed) "
                f"| {el/60:.1f}m elapsed, ~{left/60:.0f}m left")


class MatrixRunner:
    def __init__(self, out_dir: pathlib.Path, *, backends: dict[str, Any],
                 concurrency: int = 12, mech_cfg: dict[str, Any] | None = None,
                 verbose: bool = True, budget_usd: float | None = None):
        self.out = pathlib.Path(out_dir)
        (self.out / "traces").mkdir(parents=True, exist_ok=True)
        self.backends = backends
        self.concurrency = concurrency
        self.mech_cfg = mech_cfg
        self.verbose = verbose
        self.stats = RunnerStats()
        # Seeded from results already on disk so a resumed sweep does not restart
        # the budget from zero.
        self.budget = BudgetGuard.from_run_dir(self.out, budget_usd)
        self._lock = threading.Lock()
        self._compound_tc: dict[str, dict[str, int]] = {}    # world_key -> {polarity: t_c}
        self._results_fh = (self.out / "results.jsonl").open("a")

    # ---------------------------------------------------------------- world preparation
    def prepare_worlds(self, cells: list[Cell]) -> dict[str, tuple[TaskRecord, TaskRecord]]:
        """Compile every distinct world serially. Cheap, and keeps generation race-free.

        Cells that differ only in interface, model or budget share a world, so a sweep over four
        budgets and three conditions compiles each world once rather than twelve times.
        """
        import copy
        by_key: dict[str, Cell] = {}
        for c in cells:
            by_key.setdefault(c.world_key, c)

        pairs: dict[str, tuple[TaskRecord, TaskRecord]] = {}
        failures: list[str] = []
        for i, (key, c) in enumerate(sorted(by_key.items()), 1):
            cfg = copy.deepcopy(self.mech_cfg)
            # MECH_KEY is imported rather than restated: a second copy of this map is how adding
            # a mechanism raised KeyError here long after the compiler already knew about it.
            mech_key = MECH_KEY[c.mechanism]
            for k, v in c.overrides:
                # An override that matches no existing key would be silently ignored, and the
                # sweep would report a flat line for an axis it never actually varied.
                if k in cfg[mech_key]:
                    cfg[mech_key][k] = v
                elif k in cfg["background"]:
                    cfg["background"][k] = v
                else:
                    raise KeyError(
                        f"override {k!r} matches no parameter in {mech_key} or background; "
                        f"implement it in the generator before sweeping over it")
            # Worlds differing only in generator parameters must not share a filename.
            variant = stable_hash(sorted(c.overrides), length=6) if c.overrides else ""
            try:
                cp = compile_pair(cfg, c.mechanism, c.seed, confounder=c.confounder,
                                  scale=c.scale, paraphrase=c.paraphrase, variant=variant)
            except RuntimeError as exc:                 # infeasible parameter combination
                failures.append(f"{key}: {exc}")
                continue
            if not cp.invariants.ok:
                failures.append(f"{key}: invariants {cp.invariants.failures()}")
                continue
            pairs[key] = (cp.positive, cp.negative)
            # M4 is scored against the oracle closure-completion time (the anchored stage-2
            # rule); without it the width-only proxy applies and a fresh sweep would not
            # reproduce the committed rows until `scripts/rescore.py` is run.
            self._compound_tc[key] = cp.compound_tc or {}
            if self.verbose and (i % 25 == 0 or i == len(by_key)):
                print(f"  compiled {i}/{len(by_key)} worlds", flush=True)

        if failures:
            # Never silently drop cells: a missing cell is a hole in a predeclared grid and has to
            # be visible in the write-up, not inferred from a gap in a plot.
            (self.out / "world_failures.txt").write_text("\n".join(failures))
            print(f"  ! {len(failures)} world(s) could not be generated; "
                  f"see {self.out / 'world_failures.txt'}", flush=True)
        return pairs

    # ---------------------------------------------------------------- execution
    def run(self, cells: list[Cell]) -> list[RunRow]:
        pairs = self.prepare_worlds(cells)
        jobs: list[tuple[Cell, TaskRecord]] = []
        skipped = 0
        for c in cells:
            got = pairs.get(c.world_key)
            if got is None:
                skipped += 1
                continue
            jobs += [(c, got[0]), (c, got[1])]
        if skipped and self.verbose:
            print(f"  ! {skipped} cell(s) skipped: world generation failed", flush=True)

        self.stats = RunnerStats(total=len(jobs))
        if self.verbose:
            print(f"\n{len(jobs)} runs across {len(cells)} cells, concurrency={self.concurrency}",
                  flush=True)

        rows: list[RunRow] = []
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futs = {pool.submit(self._one, c, rec): (c, rec) for c, rec in jobs}
            for fut in as_completed(futs):
                c, rec = futs[fut]
                try:
                    row = fut.result()
                    if row is not None:
                        rows.append(row)
                except BudgetExceeded as exc:
                    if not getattr(self, "_budget_announced", False):
                        self._budget_announced = True
                        print(f"  !! {exc} — remaining episodes will be skipped", flush=True)
                    with self._lock:
                        self.stats.failed += 1
                except Exception as exc:                      # never let one run kill the sweep
                    with self._lock:
                        self.stats.failed += 1
                    if self.verbose:
                        print(f"  ! {c.key()} {rec.polarity}: "
                              f"{type(exc).__name__}: {str(exc)[:160]}", flush=True)
                finally:
                    with self._lock:
                        self.stats.done += 1
                        if self.verbose and self.stats.done % 10 == 0:
                            print(f"  {self.stats.line()} | {self.budget.line()}", flush=True)
        self._results_fh.flush()
        return rows

    def _one(self, cell: Cell, rec: TaskRecord) -> RunRow | None:
        backend = self.backends[cell.model_slot]
        prompt_hash = digest_text(system_prompt(cell.condition, cell.budget) + "||"
                                  + rec.task.question)
        run_id = make_run_id(rec.task.task_id, cell.condition, backend.name, prompt_hash,
                             rec.world_hash)
        path = self.out / "traces" / f"{run_id}.json"

        if path.exists():                                     # resume
            try:
                trace = Trace.from_dict(json.loads(path.read_text()))
                with self._lock:
                    self.stats.cached += 1
                return self._record(cell, rec, trace, cached=True)
            except Exception:
                path.unlink(missing_ok=True)                  # corrupt cache entry: redo it

        self.budget.reserve(getattr(backend, "name", ""))
        world = load_world(rec.world_path)
        if cell.model_slot == "script_a1":
            from agents.script_a1 import run_script_a1
            trace = run_script_a1(world, rec.task, tool_budget=cell.budget,
                                  world_hash=rec.world_hash, pair_id=rec.pair_id).trace
        else:
            trace = run_episode(task=rec.task, world=world, condition=cell.condition,
                                backend=backend, pair_id=rec.pair_id,
                                world_hash=rec.world_hash, tool_budget=cell.budget,
                                code_commit=_git_commit())
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(trace.to_dict()))
        tmp.replace(path)
        return self._record(cell, rec, trace, cached=False)

    def _record(self, cell: Cell, rec: TaskRecord, trace: Trace, *, cached: bool) -> RunRow:
        tc = self._compound_tc.get(cell.world_key, {}).get(rec.polarity)
        row = score(trace, rec, cell.condition, compound_tc=tc)
        run_cost = self.budget.record(trace.model, trace.total_input_tokens,
                                      trace.total_output_tokens)
        payload = {**{k: v for k, v in asdict(row).items() if k != "check"},
                   "check": asdict(row.check), "budget": cell.budget,
                   "model_slot": cell.model_slot, "sweep": cell.sweep, "seed": cell.seed,
                   "scale": cell.scale, "cached": cached, "run_id": trace.run_id, "cost_usd": round(run_cost, 6)}
        with self._lock:
            self._results_fh.write(json.dumps(payload) + "\n")
        return row

    # ---------------------------------------------------------------- bookkeeping
    def write_manifest(self, cells: list[Cell], extra: dict[str, Any] | None = None) -> None:
        synced = ROOT / ".synced_commit"
        (self.out / "manifest.json").write_text(json.dumps({
            "code_commit": _git_commit(),
            "synced_commit": synced.read_text().strip() if synced.exists() else None,
            # content hash of the generator config actually used — the M3 incident produced
            # results under a config that had been "fixed" only in the local checkout
            "mech_config_hash": stable_hash(self.mech_cfg) if self.mech_cfg else None,
            "n_cells": len(cells), "n_runs": 2 * len(cells),
            "models": {k: getattr(v, "name", str(v)) for k, v in self.backends.items()},
            "concurrency": self.concurrency,
            "cells": [asdict(c) for c in cells],
            **(extra or {}),
        }, indent=2))

    def close(self) -> None:
        self._results_fh.close()


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def load_results(path: pathlib.Path) -> list[dict[str, Any]]:
    """Read results.jsonl, keeping the last record per run_id (a resumed cell may appear twice)."""
    seen: dict[str, dict[str, Any]] = {}
    with pathlib.Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            seen[d.get("run_id", json.dumps(d, sort_keys=True))] = d
    return list(seen.values())
