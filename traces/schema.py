"""Trajectory schema (taskboard Part B4 / plan §26).

Every run — including smoke tests — writes one of these (R12). They are the evidence base for
claim checking (cited ``step_NN`` ids must exist here), the cost analysis, the error taxonomy, and
later the workload-characterisation paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import SchemaError, check_enum, check_type, require_keys, stable_hash

AGENT_CONDITIONS = ("A0", "A1", "A2", "A3", "A4")
RUN_STATUSES = (
    "success",           # agent returned a schema-valid claim
    "malformed_output",  # two consecutive schema failures
    "budget_exhausted",  # hit the tool-call ceiling without a claim
    "tool_error",        # unrecoverable tool failure
    "api_error",         # transport failure after the permitted retry
    "timeout",
)


@dataclass
class Step:
    step_id: str                 # "step_01" — the id an agent cites as evidence
    tool: str
    args: dict[str, Any]
    result_digest: str           # hash of the full result (kept out of the trace for size)
    result_summary: str          # short human-readable form, for error analysis
    latency_ms: int
    tokens_before: int
    tokens_after: int
    ok: bool = True
    error: str = ""
    rows_scanned: int = 0        # C6: events the backend examined for this call

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows_scanned": self.rows_scanned,
            "step_id": self.step_id, "tool": self.tool, "args": self.args,
            "result_digest": self.result_digest, "result_summary": self.result_summary,
            "latency_ms": self.latency_ms, "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after, "ok": self.ok, "error": self.error,
        }


@dataclass
class Trace:
    run_id: str
    task_id: str
    pair_id: str                 # internal bookkeeping; traces are not served to agents
    world_hash: str
    agent_condition: str
    model: str                   # exact pinned version string
    prompt_hash: str
    code_commit: str
    tool_budget: int
    steps: list[Step] = field(default_factory=list)
    final_claim: dict[str, Any] | None = None
    raw_final_response: str = ""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_ms: int = 0
    llm_calls: int = 0
    schema_repairs: int = 0
    tool_protocol_errors: int = 0    # malformed tool calls (C4: not analytical failure)
    status: str = "success"
    error: str = ""

    # -- helpers ------------------------------------------------------------------------
    def next_step_id(self) -> str:
        return f"step_{len(self.steps) + 1:02d}"

    def step_ids(self) -> set[str]:
        return {s.step_id for s in self.steps}

    def step(self, step_id: str) -> Step | None:
        for s in self.steps:
            if s.step_id == step_id:
                return s
        return None

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "task_id": self.task_id, "pair_id": self.pair_id,
            "world_hash": self.world_hash, "agent_condition": self.agent_condition,
            "model": self.model, "prompt_hash": self.prompt_hash, "code_commit": self.code_commit,
            "tool_budget": self.tool_budget, "steps": [s.to_dict() for s in self.steps],
            "final_claim": self.final_claim, "raw_final_response": self.raw_final_response,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens, "total_latency_ms": self.total_latency_ms,
            "llm_calls": self.llm_calls, "schema_repairs": self.schema_repairs,
            "tool_protocol_errors": self.tool_protocol_errors,
            "status": self.status, "error": self.error,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Trace":
        validate_trace(d)
        t = Trace(
            run_id=d["run_id"], task_id=d["task_id"], pair_id=d["pair_id"],
            world_hash=d["world_hash"], agent_condition=d["agent_condition"], model=d["model"],
            prompt_hash=d["prompt_hash"], code_commit=d.get("code_commit", ""),
            tool_budget=int(d["tool_budget"]), final_claim=d.get("final_claim"),
            raw_final_response=d.get("raw_final_response", ""),
            total_input_tokens=int(d.get("total_input_tokens", 0)),
            total_output_tokens=int(d.get("total_output_tokens", 0)),
            total_latency_ms=int(d.get("total_latency_ms", 0)),
            llm_calls=int(d.get("llm_calls", 0)),
            schema_repairs=int(d.get("schema_repairs", 0)),
            tool_protocol_errors=int(d.get("tool_protocol_errors", 0)),
            status=d.get("status", "success"), error=d.get("error", ""),
        )
        t.steps = [Step(**{k: v for k, v in s.items()}) for s in d.get("steps", [])]
        return t


def validate_trace(d: dict[str, Any]) -> None:
    where = "trace"
    check_type(d, dict, where)
    require_keys(d, ("run_id", "task_id", "pair_id", "world_hash", "agent_condition", "model",
                     "prompt_hash", "tool_budget", "steps"), where)
    check_enum(d["agent_condition"], AGENT_CONDITIONS, f"{where}.agent_condition")
    check_enum(d.get("status", "success"), RUN_STATUSES, f"{where}.status")
    check_type(d["steps"], list, f"{where}.steps")

    seen: set[str] = set()
    for i, s in enumerate(d["steps"]):
        w = f"{where}.steps[{i}]"
        check_type(s, dict, w)
        require_keys(s, ("step_id", "tool", "args", "result_digest"), w)
        if s["step_id"] in seen:
            raise SchemaError(f"{w}.step_id: duplicate {s['step_id']!r}")
        seen.add(s["step_id"])
    expected = {f"step_{i:02d}" for i in range(1, len(d["steps"]) + 1)}
    if seen != expected:
        raise SchemaError(f"{where}.steps: ids must be contiguous step_01..step_NN, got "
                          f"{sorted(seen)}")


def make_run_id(task_id: str, agent_condition: str, model: str, prompt_hash: str,
                world_hash: str) -> str:
    """Deterministic run id — doubles as the cache key for the resumable runner (WP8)."""
    return "r_" + stable_hash({"t": task_id, "a": agent_condition, "m": model,
                               "p": prompt_hash, "w": world_hash}, length=12)
