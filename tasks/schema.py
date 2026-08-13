"""Task schema (taskboard Part B1).

Two objects, deliberately separated:

* ``Task``       — what the agent sees. Carries no pair id, no polarity, no mechanism label, no
                   gold. Its ``task_id`` is a content hash of a private salt so that neighbouring
                   ids reveal nothing about pairing (R5, blinding).
* ``TaskRecord`` — internal bookkeeping: the served task plus pair id, polarity, mechanism and gold
                   label. Never written into any directory the agent can read.

``scripts/leakage_audit.py`` enforces the separation by scanning served artifacts for
``FORBIDDEN_SERVED_KEYS``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import (SchemaError, check_enum, check_interval, check_type, forbid_keys,
                    require_keys, stable_hash)

MECHANISMS = ("cascade", "closure_wave", "regime_switch", "compound")
POLARITIES = ("positive", "negative")

#: keys that must never appear anywhere in a served task/world artifact
FORBIDDEN_SERVED_KEYS = (
    "gold", "gold_label", "label", "polarity", "positive", "negative", "pair_id",
    "mechanism", "mechanism_params", "planted", "planted_nodes", "planted_edges",
    "invariants", "margin", "seed", "confounder", "confounder_level", "answer",
)

OUTPUT_CONTRACT_ID = "claim.v1"


@dataclass(frozen=True)
class Task:
    """The served task. Everything here is safe for the agent to see."""

    task_id: str
    question: str
    horizon: tuple[int, int]
    focus: dict[str, Any]
    output_contract: str = OUTPUT_CONTRACT_ID

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "question": self.question,
            "horizon": list(self.horizon),
            "focus": dict(self.focus),
            "output_contract": self.output_contract,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Task":
        validate_task(d)
        return Task(
            task_id=d["task_id"],
            question=d["question"],
            horizon=tuple(d["horizon"]),  # type: ignore[arg-type]
            focus=dict(d["focus"]),
            output_contract=d["output_contract"],
        )


#: focus keys permitted per mechanism — anything else is a leak or a typo
ALLOWED_FOCUS_KEYS = {
    "cascade": {"source", "target", "window", "max_wait"},
    "closure_wave": {"nodes", "window", "delta"},
    # threshold added by the third M3 amendment: the decision cut is task specification, not gold,
    # and the served question states it explicitly.
    "regime_switch": {"group", "window", "bandwidth", "threshold"},
}
_ANY_FOCUS_KEY = set().union(*ALLOWED_FOCUS_KEYS.values())


def validate_task(d: dict[str, Any]) -> None:
    where = "task"
    check_type(d, dict, where)
    require_keys(d, ("task_id", "question", "horizon", "focus", "output_contract"), where)
    forbid_keys(d, FORBIDDEN_SERVED_KEYS, where)

    check_type(d["task_id"], str, f"{where}.task_id")
    if not d["task_id"].startswith("t_"):
        raise SchemaError(f"{where}.task_id: expected 't_' prefix, got {d['task_id']!r}")

    check_type(d["question"], str, f"{where}.question")
    if not d["question"].strip():
        raise SchemaError(f"{where}.question: empty")

    lo, hi = check_interval(d["horizon"], f"{where}.horizon")

    check_type(d["focus"], dict, f"{where}.focus")
    forbid_keys(d["focus"], FORBIDDEN_SERVED_KEYS, f"{where}.focus")
    unknown = set(d["focus"]) - _ANY_FOCUS_KEY
    if unknown:
        raise SchemaError(f"{where}.focus: unknown key(s) {sorted(unknown)}")
    if "window" in d["focus"]:
        wlo, whi = check_interval(d["focus"]["window"], f"{where}.focus.window")
        if wlo < lo or whi > hi:
            raise SchemaError(f"{where}.focus.window {[wlo, whi]} outside horizon {[lo, hi]}")

    check_enum(d["output_contract"], (OUTPUT_CONTRACT_ID,), f"{where}.output_contract")


@dataclass(frozen=True)
class TaskRecord:
    """Internal record. Never served."""

    task: Task
    pair_id: str
    polarity: str          # positive | negative
    mechanism: str         # cascade | closure_wave | regime_switch
    gold: bool             # True iff the mechanism is genuinely present
    world_path: str
    world_hash: str
    confounder: str        # "none" | "c1_low" | ...
    seed: int
    paraphrase: int = 1

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["task"] = self.task.to_dict()
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "TaskRecord":
        validate_task_record(d)
        return TaskRecord(
            task=Task.from_dict(d["task"]), pair_id=d["pair_id"], polarity=d["polarity"],
            mechanism=d["mechanism"], gold=bool(d["gold"]), world_path=d["world_path"],
            world_hash=d["world_hash"], confounder=d["confounder"], seed=int(d["seed"]),
            paraphrase=int(d.get("paraphrase", 1)),
        )


def validate_task_record(d: dict[str, Any]) -> None:
    where = "task_record"
    check_type(d, dict, where)
    require_keys(d, ("task", "pair_id", "polarity", "mechanism", "gold", "world_path",
                     "world_hash", "confounder", "seed"), where)
    validate_task(d["task"])
    check_enum(d["polarity"], POLARITIES, f"{where}.polarity")
    check_enum(d["mechanism"], MECHANISMS, f"{where}.mechanism")
    check_type(d["gold"], bool, f"{where}.gold")
    if (d["polarity"] == "positive") != bool(d["gold"]):
        raise SchemaError(f"{where}: polarity {d['polarity']!r} contradicts gold {d['gold']!r}")
    check_type(d["seed"], int, f"{where}.seed")
    unknown = set(d["task"]["focus"]) - ALLOWED_FOCUS_KEYS[d["mechanism"]]
    if unknown:
        raise SchemaError(
            f"{where}.task.focus: key(s) {sorted(unknown)} not allowed for {d['mechanism']}")


def make_task_id(pair_id: str, polarity: str, paraphrase: int, salt: str = "atgp-v1") -> str:
    """Opaque, deterministic, order-revealing-nothing task id (R5).

    Hashing with a salt means consecutive ids look unrelated, so an agent cannot infer that two
    tasks form a pair, nor which member it is looking at.
    """
    return "t_" + stable_hash({"p": pair_id, "q": polarity, "n": paraphrase, "s": salt}, length=8)
