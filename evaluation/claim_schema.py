"""Claim schema (taskboard Part B2) — the agent's machine-checkable output.

The narrative field exists but is never scored for correctness (plan §12). Primary labels come from
this structured object alone, so no result depends on an LLM judge.

Scoring convention (taskboard R3): gold is binary {present, absent}. ``insufficient`` is therefore
scored as **incorrect** and tracked separately as an abstention.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import (SchemaError, check_enum, check_int_list, check_interval, check_type,
                    check_unit, require_keys)

DECISIONS = ("present", "absent", "insufficient")
# Imported, not restated. Two independent copies of this tuple existed, and neither learned
# about M4 when it was added, so every compound claim failed validation and 30 episodes reported
# `malformed_output` -- a harness defect that would have read as "the model cannot do compound
# reasoning". One definition, in the schema that owns the concept.
from tasks.schema import MECHANISMS  # noqa: E402
EVIDENCE_ROLES = ("support", "confounder_control")

_STEP_RE = re.compile(r"^step_\d{2,3}$")


@dataclass(frozen=True)
class Evidence:
    trace_id: str
    role: str

    def to_dict(self) -> dict[str, str]:
        return {"trace_id": self.trace_id, "role": self.role}


@dataclass(frozen=True)
class Claim:
    task_id: str
    decision: str
    mechanism: str
    entities: list[int]
    window: tuple[int, int]
    confidence: float
    evidence: list[Evidence]
    narrative: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "decision": self.decision,
            "mechanism": self.mechanism,
            "entities": list(self.entities),
            "window": list(self.window),
            "confidence": self.confidence,
            "evidence": [e.to_dict() for e in self.evidence],
            "narrative": self.narrative,
        }

    # -- convenience predicates used by the scorer -------------------------------------
    @property
    def is_positive(self) -> bool:
        return self.decision == "present"

    @property
    def is_abstention(self) -> bool:
        return self.decision == "insufficient"

    def cited(self, role: str) -> list[str]:
        return [e.trace_id for e in self.evidence if e.role == role]

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Claim":
        validate_claim(d)
        return Claim(
            task_id=d["task_id"], decision=d["decision"], mechanism=d["mechanism"],
            entities=check_int_list(d["entities"], "claim.entities"),
            window=check_interval(d["window"], "claim.window"),
            confidence=float(d["confidence"]),
            evidence=[Evidence(e["trace_id"], e["role"]) for e in d["evidence"]],
            narrative=str(d.get("narrative", "")),
        )


def validate_claim(d: dict[str, Any]) -> None:
    where = "claim"
    check_type(d, dict, where)
    require_keys(d, ("task_id", "decision", "mechanism", "entities", "window", "confidence",
                     "evidence"), where)

    check_type(d["task_id"], str, f"{where}.task_id")
    check_enum(d["decision"], DECISIONS, f"{where}.decision")
    check_enum(d["mechanism"], MECHANISMS, f"{where}.mechanism")
    check_int_list(d["entities"], f"{where}.entities")
    check_interval(d["window"], f"{where}.window")
    check_unit(d["confidence"], f"{where}.confidence")

    check_type(d["evidence"], list, f"{where}.evidence")
    seen = set()
    for i, e in enumerate(d["evidence"]):
        w = f"{where}.evidence[{i}]"
        check_type(e, dict, w)
        require_keys(e, ("trace_id", "role"), w)
        check_type(e["trace_id"], str, f"{w}.trace_id")
        if not _STEP_RE.match(e["trace_id"]):
            raise SchemaError(f"{w}.trace_id: expected 'step_NN', got {e['trace_id']!r}")
        check_enum(e["role"], EVIDENCE_ROLES, f"{w}.role")
        key = (e["trace_id"], e["role"])
        if key in seen:
            raise SchemaError(f"{w}: duplicate citation {key}")
        seen.add(key)

    if "narrative" in d:
        check_type(d["narrative"], str, f"{where}.narrative")

    # A positive claim asserting nothing about who or when is not scoreable.
    if d["decision"] == "present" and not d["entities"]:
        raise SchemaError(f"{where}: decision 'present' requires a non-empty entities list")


# --------------------------------------------------------------------------------------
# tolerant extraction from a model response
# --------------------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_claim(text: str) -> dict[str, Any]:
    """Pull the claim object out of a model response.

    Accepts a bare JSON object, a fenced block, or a trailing object after prose. Raises
    SchemaError if nothing parseable is present — the harness then issues exactly one repair
    retry (taskboard T1.6), and a second failure is recorded as a malformed-output run.
    """
    candidates: list[str] = []
    stripped = text.strip()
    if stripped.startswith("{"):
        candidates.append(stripped)
    candidates.extend(m.group(1) for m in _FENCE_RE.finditer(text))
    candidates.extend(_balanced_objects(text))

    best_err: Exception | None = None
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError as exc:
            best_err = exc
            continue
        if isinstance(obj, dict) and "decision" in obj:
            return obj
    raise SchemaError(f"no claim JSON found in response ({best_err})")


def _balanced_objects(text: str) -> list[str]:
    """Every balanced {...} span, longest first — tolerates prose on both sides."""
    out, starts = [], []
    for i, ch in enumerate(text):
        if ch == "{":
            starts.append(i)
        elif ch == "}" and starts:
            out.append(text[starts.pop():i + 1])
    return sorted(out, key=len, reverse=True)
