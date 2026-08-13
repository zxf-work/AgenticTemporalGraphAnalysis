"""World -> task compiler.

Produces, per (mechanism, confounder, seed):

    data/worlds/<pair_id>_pos.csv      served event table (u, v, t) — no metadata
    data/worlds/<pair_id>_neg.csv
    data/tasks/<pair_id>.json          two TaskRecords (internal: gold, polarity, pair id)
    data/invariants/<pair_id>.json     the invariant record

The served CSV and the served half of each TaskRecord contain nothing that identifies polarity,
mechanism, or planted structure; ``scripts/leakage_audit.py`` enforces this.
"""

from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Any, Callable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tasks.schema import Task, TaskRecord, make_task_id
from tasks.templates import render
from worlds.base import TemporalWorld
from worlds.invariants import InvariantRecord, check_pair, static_view_fingerprint
from worlds.mechanisms.cascade import (cascade_global_check, cascade_gold, generate_cascade_pair)
from worlds.mechanisms.closure_wave import (closure_global_check, closure_gold,
                                            generate_closure_pair)
from worlds.mechanisms.compound import (closure_completion_time, compound_global_check,
                                        compound_gold, generate_compound_pair)
from worlds.mechanisms.regime_switch import (generate_regime_pair, regime_global_check,
                                             regime_gold)

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"

MECH_KEY = {"cascade": "m1_cascade", "closure_wave": "m2_closure_wave",
            "regime_switch": "m3_regime_switch", "compound": "m4_compound"}
MECH_TAG = {"cascade": "m1", "closure_wave": "m2", "regime_switch": "m3",
            "compound": "m4"}


@dataclass
class CompiledPair:
    pair_id: str
    positive: TaskRecord
    negative: TaskRecord
    invariants: InvariantRecord
    # Ground-truth closure-completion time per polarity, compound only. Never served to the
    # agent; consumed by the M4 evidence checker, which verifies that a cited stage-2 query is
    # anchored at the true t_c rather than merely narrow (evaluation/error_taxonomy.py).
    compound_tc: dict[str, int] | None = None


def _focus_for(mechanism: str, spec: Any) -> dict[str, Any]:
    """The scope the agent is legitimately told. Never includes gold or planted structure."""
    win = [int(spec.window[0]), int(spec.window[1])]
    if mechanism == "cascade":
        return {"source": int(spec.source), "target": int(spec.target), "window": win,
                "max_wait": int(spec.max_wait)}
    if mechanism == "closure_wave":
        # the node set is told (it scopes the question) but not which triples form triangles
        return {"nodes": [int(n) for n in spec.nodes], "window": win, "delta": int(spec.delta)}
    if mechanism == "regime_switch":
        # the focal group is told; the old/new partner sets and the change point are not
        return {"group": [int(n) for n in spec.group], "window": win,
                "bandwidth": int(spec.bandwidth)}
    if mechanism == "compound":
        # Everything the agent legitimately needs to run BOTH stages and bind them: the group and
        # closure bound for stage 1, the endpoints and waiting bound for stage 2, and the
        # admissible lag that ties the second to the first. t_c itself is NOT given -- computing
        # it is the compositional step under test.
        return {"group": [int(n) for n in spec.group], "window": win, "delta": int(spec.delta),
                "source": int(spec.source), "target": int(spec.target),
                "max_wait": int(spec.max_wait), "admissible": int(spec.admissible)}
    raise NotImplementedError(f"focus for {mechanism}")


def _gold_fns(mechanism: str, spec: Any) -> tuple[Callable, Callable]:
    if mechanism == "cascade":
        kw = dict(source=spec.source, target=spec.target, window=spec.window,
                  max_wait=spec.max_wait)
        return (lambda w: cascade_gold(w, **kw)), (lambda w: cascade_global_check(w, **kw))
    if mechanism == "closure_wave":
        kw = dict(nodes=spec.nodes, window=spec.window, delta=spec.delta,
                  threshold=spec.threshold)
        return (lambda w: closure_gold(w, **kw)), (lambda w: closure_global_check(w, **kw))
    if mechanism == "regime_switch":
        kw = dict(group=spec.group, window=spec.window, bandwidth=spec.bandwidth,
                  threshold=spec.threshold)
        return (lambda w: regime_gold(w, **kw)), (lambda w: regime_global_check(w, **kw))
    if mechanism == "compound":
        kw = dict(group=spec.group, window=spec.window, delta=spec.delta,
                  closure_threshold=spec.closure_threshold, source=spec.source,
                  target=spec.target, max_wait=spec.max_wait, admissible=spec.admissible)
        return (lambda w: compound_gold(w, **kw)), (lambda w: compound_global_check(w, **kw))
    raise NotImplementedError(f"gold for {mechanism}")


GENERATORS: dict[str, Callable] = {"cascade": generate_cascade_pair,
                                   "closure_wave": generate_closure_pair,
                                   "regime_switch": generate_regime_pair,
                                   "compound": generate_compound_pair}


def compile_pair(cfg: dict[str, Any], mechanism: str, seed: int, *, confounder: str = "none",
                 scale: str = "dev", paraphrase: int = 1, out_root: pathlib.Path | None = None,
                 write: bool = True, variant: str = "") -> CompiledPair:
    """Compile one matched pair.

    ``variant`` distinguishes worlds that share (mechanism, confounder, seed) but were generated
    with different parameters — different path lengths, decoy counts or temporal resolutions.
    Without it a sweep over a generator axis silently overwrites its own worlds on disk and every
    cell ends up reading whichever variant was compiled last, so the axis appears to have no
    effect. Callers that vary generator parameters MUST pass it.
    """
    root = out_root or DATA
    pos_w, neg_w, spec, attempts = GENERATORS[mechanism](cfg, seed, scale=scale)

    # confounder overlay is applied identically to both worlds (plan §8); "none" is a no-op
    if confounder != "none":
        from worlds.confounders import apply_confounder      # local import: optional module
        pos_w, neg_w = apply_confounder(confounder, pos_w, neg_w, cfg, seed)

    m_cfg = cfg[MECH_KEY[mechanism]]
    gold_fn, global_fn = _gold_fns(mechanism, spec)
    pair_id = (f"{MECH_TAG[mechanism]}_{confounder}_seed{seed:03d}"
               + (f"_{variant}" if variant else ""))

    focus = _focus_for(mechanism, spec)
    window = (focus["window"][0], focus["window"][1])
    focus_nodes = (focus.get("nodes") or focus.get("group")
                   or [focus.get("source"), focus.get("target")])

    inv = check_pair(pos_w, neg_w, pair_id=pair_id, mechanism=mechanism, confounder=confounder,
                     seed=seed, gold_fn=gold_fn, required_margin=m_cfg["required_margin"],
                     threshold=m_cfg["threshold"], global_check_fn=global_fn,
                     static_fingerprint_fn=lambda w: static_view_fingerprint(w, focus_nodes,
                                                                            window),
                     rejection_attempts=attempts)
    records = {}
    for polarity, world in (("positive", pos_w), ("negative", neg_w)):
        tid = make_task_id(pair_id, polarity, paraphrase)
        rel = f"data/worlds/{pair_id}_{polarity[:3]}.csv"
        if write:
            world.write_served(root.parent / rel)
        task = Task(task_id=tid,
                    question=render(mechanism, paraphrase, task_id=tid, focus=focus,
                                    window=window),
                    horizon=tuple(world.horizon), focus=focus)
        records[polarity] = TaskRecord(
            task=task, pair_id=pair_id, polarity=polarity, mechanism=mechanism,
            gold=(polarity == "positive"), world_path=rel, world_hash=world.world_hash(),
            confounder=confounder, seed=seed, paraphrase=paraphrase)

    if write:
        (root / "tasks").mkdir(parents=True, exist_ok=True)
        (root / "invariants").mkdir(parents=True, exist_ok=True)
        (root / "tasks" / f"{pair_id}.json").write_text(json.dumps(
            {"pair_id": pair_id,
             "positive": records["positive"].to_dict(),
             "negative": records["negative"].to_dict()}, indent=2))
        (root / "invariants" / f"{pair_id}.json").write_text(
            json.dumps(inv.to_dict(), indent=2))

    compound_tc = None
    if mechanism == "compound":
        compound_tc = {}
        for polarity, world in (("positive", pos_w), ("negative", neg_w)):
            tc = closure_completion_time(world, group=spec.group, window=tuple(spec.window),
                                         delta=spec.delta)
            if tc is not None:
                compound_tc[polarity] = int(tc)

    return CompiledPair(pair_id=pair_id, positive=records["positive"],
                        negative=records["negative"], invariants=inv, compound_tc=compound_tc)


def load_world(rel_path: str) -> TemporalWorld:
    """Reload a served world exactly as the agent sees it: (u, v, t) plus the neutral sidecar.

    Every event is tagged ``background`` on reload — provenance is not recoverable from the served
    files, which is the point. Domain facts (node count, horizon, directedness) come from the
    sidecar so the tool layer computes over the same domain as the gold oracle.
    """
    import csv
    import json
    from worlds.base import TAG_BACKGROUND
    p = pathlib.Path(__file__).resolve().parents[1] / rel_path
    meta = json.loads(p.with_suffix(".meta.json").read_text())
    rows = []
    with p.open() as fh:
        for r in csv.DictReader(fh):
            rows.append((int(r["u"]), int(r["v"]), int(r["t"]), TAG_BACKGROUND))
    return TemporalWorld.from_events(int(meta["n_nodes"]), rows, tuple(meta["horizon"]),
                                     directed=bool(meta["directed"]))
