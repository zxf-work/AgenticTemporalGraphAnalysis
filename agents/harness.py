"""Agent harness — one loop, several backends, several tool surfaces.

Design rule: **the loop is identical for every condition.** A1, A3 and A4 differ only in the tool
surface they are given and in one paragraph of the system prompt. If the loop itself varied, an
A3-over-A1 result would be a statement about harness engineering rather than about ordering-aware
analysis (plan §13, H2).

Backends implement one method, so a scripted `MockBackend` can drive the entire pipeline — tool
dispatch, tracing, claim extraction, repair retry, budget enforcement — without any API key. The
mock also implements the sanity predictors required by R4.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agents.tools_spec import ToolSpec, surface_for
from common import digest_text, stable_hash
from evaluation.claim_schema import SchemaError, extract_claim, validate_claim
from tasks.schema import Task
from traces.schema import Step, Trace, make_run_id
from worlds.base import TemporalWorld

MAX_RESULT_CHARS = 4000          # bounded evidence: identical cap for every condition


# ======================================================================================
# backend protocol
# ======================================================================================

@dataclass
class LLMReply:
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)   # [{name, args, id}]
    input_tokens: int = 0
    output_tokens: int = 0
    # Tool-protocol failures: the model emitted something in the tool-call slot that could not be
    # read as a call (unparseable JSON arguments, a name that is not an offered tool). Open-weight
    # models do this far more than the hosted ones, and C4 requires that such a failure is never
    # scored as an analytical failure -- an agent that could not emit a well-formed call has not
    # been measured on its temporal reasoning at all.
    protocol_errors: list[str] = field(default_factory=list)


class Backend(Protocol):
    name: str

    def complete(self, system: str, messages: list[dict[str, Any]],
                 tools: list[ToolSpec]) -> LLMReply: ...


# ======================================================================================
# system prompts — the ONLY textual difference between conditions
# ======================================================================================

BASE_SYSTEM = """You are a data analyst working with a temporal event graph.

The graph is a list of interactions (u, v, t): node u interacted with node v at time t. You cannot
see the graph directly. You inspect it by calling the tools provided.

Work step by step. Call tools to gather evidence, then state a conclusion. You have a budget of
{budget} tool calls; spend them purposefully.

When you are ready, reply with the JSON object described in the task. Every trace_id you cite in
the evidence list must be a step you actually performed in this session; each tool result is
labelled with its step id."""

CONTRASTIVE_CLAUSE = """

REPORTING REQUIREMENT. A conclusion of "present" is only acceptable when you can cite both:
  (a) at least one tool result that directly supports the mechanism, with role "support"; and
  (b) at least one tool result that tests a competing explanation — for instance that the pattern
      is explained by overall activity volume, by the popularity of the nodes involved, by
      repeated interactions, or by structure that ignores timing — with role "confounder_control".

If you have not yet gathered both, and you still have tool calls left, gather them before
concluding. If your budget is spent and you still lack one of them, answer "insufficient" rather
than asserting a mechanism you cannot support."""


CODE_CLAUSE = """

You have a Python interpreter with the complete event log in the variable `events`, a list of
(u, v, t) tuples sorted by t. There are no analysis tools: you write the analysis yourself. Each
call runs in a fresh interpreter with no memory of previous calls, so every program must be
self-contained, and only what you print comes back to you."""


def system_prompt(condition: str, budget: int) -> str:
    p = BASE_SYSTEM.format(budget=budget)
    if condition == "A4":
        p += CONTRASTIVE_CLAUSE
    if condition == "A2":
        # A2 differs from A1/A3 in exactly one paragraph, as they differ from each other, so the
        # comparison stays a statement about the interface rather than about prompt engineering.
        p += CODE_CLAUSE
    return p


# ======================================================================================
# the loop
# ======================================================================================

def run_episode(*, task: Task, world: TemporalWorld, condition: str, backend: Backend,
                pair_id: str = "", world_hash: str = "", tool_budget: int = 20,
                code_commit: str = "", max_turns: int | None = None) -> Trace:
    """Run one analytical episode and return its full trajectory."""
    tools = surface_for(condition)
    sysmsg = system_prompt(condition, tool_budget)
    prompt_hash = digest_text(sysmsg + "||" + task.question)
    trace = Trace(
        run_id=make_run_id(task.task_id, condition, backend.name, prompt_hash,
                           world_hash or world.world_hash()),
        task_id=task.task_id, pair_id=pair_id, world_hash=world_hash or world.world_hash(),
        agent_condition=condition, model=backend.name, prompt_hash=prompt_hash,
        code_commit=code_commit, tool_budget=tool_budget)

    messages: list[dict[str, Any]] = [{"role": "user", "content": task.question}]
    by_name = {t.name: t for t in tools}
    turns = max_turns or (tool_budget + 4)
    t_start = time.time()

    for _turn in range(turns):
        reply = _call(backend, sysmsg, messages, tools, trace)
        if reply is None:
            trace.status = "api_error"
            break

        if reply.protocol_errors:
            # The model put something in the tool-call slot that could not be read as a call.
            # Count it, tell the model plainly what was wrong, and let it retry -- but never
            # charge it to the tool budget, which measures analysis, not syntax. Open-weight
            # subjects need this; the hosted ones almost never trigger it.
            trace.tool_protocol_errors += len(reply.protocol_errors)
            if not reply.tool_calls:
                messages.append({"role": "assistant", "content": reply.text or ""})
                messages.append({"role": "user", "content":
                                 "Your last tool call could not be parsed ("
                                 + "; ".join(reply.protocol_errors[:3])
                                 + "). Re-issue it as a single well-formed tool call whose "
                                   "arguments are a JSON object matching the tool's schema."})
                continue

        if reply.tool_calls:
            if len(trace.steps) >= tool_budget:
                messages.append({"role": "user", "content":
                                 "Tool budget exhausted. Give your final JSON conclusion now, "
                                 "citing only steps you already performed."})
                continue
            messages.append({"role": "assistant", "content": reply.text or "",
                             "tool_calls": reply.tool_calls})
            # Every declared tool_call must receive a reply, including ones refused for budget:
            # a dangling tool_call_id is a protocol error that some providers reject outright,
            # which would otherwise show up as a spurious model failure in the results.
            for call in reply.tool_calls:
                if len(trace.steps) >= tool_budget:
                    messages.append({
                        "role": "tool", "tool_call_id": call.get("id", ""),
                        "name": call["name"],
                        "content": "Tool budget exhausted; this call was not executed."})
                    continue
                step = _dispatch(call, by_name, world, trace)
                messages.append({"role": "tool", "tool_call_id": call.get("id", step.step_id),
                                 "name": call["name"],
                                 "content": f"[{step.step_id}] {step.result_summary}"})
            continue

        # no tool calls -> the model is concluding
        claim = _parse_claim(reply.text, task, trace)
        if claim is not None:
            trace.final_claim = claim
            trace.raw_final_response = reply.text[:4000]
            trace.status = "success"
            break
        if trace.schema_repairs > 1:
            trace.status = "malformed_output"
            trace.raw_final_response = reply.text[:4000]
            break
        messages.append({"role": "assistant", "content": reply.text})
        messages.append({"role": "user", "content":
                         "That was not a valid conclusion object. Reply with ONLY the JSON object "
                         "specified in the task, with all required fields."})
    else:
        trace.status = "budget_exhausted"

    trace.total_latency_ms = int((time.time() - t_start) * 1000)
    return trace


def _call(backend: Backend, sysmsg: str, messages: list[dict[str, Any]], tools: list[ToolSpec],
          trace: Trace) -> LLMReply | None:
    try:
        reply = backend.complete(sysmsg, messages, tools)
    except Exception as exc:                       # transport failure after backend's own retry
        trace.error = f"{type(exc).__name__}: {exc}"[:500]
        return None
    trace.llm_calls += 1
    trace.total_input_tokens += reply.input_tokens
    trace.total_output_tokens += reply.output_tokens
    return reply


def _dispatch(call: dict[str, Any], by_name: dict[str, ToolSpec], world: TemporalWorld,
              trace: Trace) -> Step:
    name, args = call["name"], dict(call.get("args") or {})
    step_id = trace.next_step_id()
    t0 = time.time()
    tokens_before = trace.total_input_tokens

    spec = by_name.get(name)
    if spec is None:
        step = Step(step_id=step_id, tool=name, args=args, result_digest="",
                    result_summary=f"ERROR: no tool named {name!r} is available to you.",
                    latency_ms=0, tokens_before=tokens_before, tokens_after=tokens_before,
                    ok=False, error="unknown_tool")
        trace.steps.append(step)
        return step

    scanned_before = world.scanned
    try:
        result = spec.fn(world, **args)
        summary = _shape(result)
        digest = stable_hash(result)
        ok, err = True, ""
    except Exception as exc:
        result, digest = None, ""
        summary = f"ERROR: {type(exc).__name__}: {exc}"
        ok, err = False, f"{type(exc).__name__}"

    rows = world.scanned - scanned_before
    if name == "run_python" and isinstance(result, dict):
        # A2 hands the entire event log to the subprocess, so the scan is the whole log per call
        # regardless of what the program then does with it. Counting only the world-side passes
        # would report A2's backend cost as zero and make its low call count look free.
        rows = max(rows, int(result.get("n_events", 0)))
    step = Step(step_id=step_id, tool=name, args=args, result_digest=digest, rows_scanned=rows,
                result_summary=summary, latency_ms=int((time.time() - t0) * 1000),
                tokens_before=tokens_before, tokens_after=trace.total_input_tokens,
                ok=ok, error=err)
    trace.steps.append(step)
    return step


def _shape(result: Any) -> str:
    """Bounded, deterministic rendering of a tool result (identical cap for all conditions)."""
    text = json.dumps(result, sort_keys=True, default=str)
    if len(text) <= MAX_RESULT_CHARS:
        return text
    return text[:MAX_RESULT_CHARS] + f"... [truncated, {len(text)} chars total]"


def _parse_claim(text: str, task: Task, trace: Trace) -> dict[str, Any] | None:
    try:
        obj = extract_claim(text)
    except SchemaError:
        trace.schema_repairs += 1
        return None
    obj.setdefault("task_id", task.task_id)
    obj["task_id"] = task.task_id            # never trust the model to echo the id
    obj.setdefault("narrative", "")
    obj.setdefault("evidence", [])
    obj.setdefault("entities", [])
    obj.setdefault("window", list(task.horizon))
    obj.setdefault("confidence", 0.5)
    obj.setdefault("mechanism", _mech_from_task(task))
    try:
        validate_claim(obj)
    except SchemaError:
        trace.schema_repairs += 1
        return None
    return obj


def _mech_from_task(task: Task) -> str:
    f = task.focus
    if "source" in f:
        return "cascade"
    if "delta" in f or "nodes" in f:
        return "closure_wave"
    return "regime_switch"


# ======================================================================================
# mock backend — drives the whole pipeline with no API key, and provides the R4 predictors
# ======================================================================================

class MockBackend:
    """Scripted backend.

    ``policy`` selects behaviour:

    * ``always_present`` / ``always_absent``  — degenerate predictors (R4 floors)
    * ``coin_flip``                           — seeded 50/50 predictor (R4 floor ~25% pair acc)
    * ``static_oracle``                       — answers from order-blind information only; must
                                                score 0 pair accuracy by construction
    * ``tool_then_answer``                    — calls a scripted tool sequence, then answers using
                                                the tool result: exercises the full loop
    * ``malformed``                           — returns prose, to exercise the repair path
    """

    def __init__(self, policy: str = "tool_then_answer", seed: int = 0,
                 script: list[dict[str, Any]] | None = None):
        self.name = f"mock:{policy}"
        self.policy = policy
        self.seed = seed
        self.script = script or []
        self._turn = 0

    def complete(self, system: str, messages: list[dict[str, Any]],
                 tools: list[ToolSpec]) -> LLMReply:
        self._turn += 1
        task_text = messages[0]["content"]
        tid = _task_id_from(task_text)

        if self.policy == "malformed":
            return LLMReply(text="I believe something happened, but I cannot be precise.",
                            input_tokens=100, output_tokens=20)

        # the script is orthogonal to the decision policy: it says which tools to call, the
        # policy says what to conclude afterwards
        if self._turn <= len(self.script):
            call = self.script[self._turn - 1]
            return LLMReply(text="", tool_calls=[{**call, "id": f"c{self._turn}"}],
                            input_tokens=200, output_tokens=30)

        decision = self._decide(tid, messages, tools)
        evidence = []
        for m in messages:
            if m.get("role") == "tool":
                sid = m["content"].split("]")[0].lstrip("[")
                evidence.append({"trace_id": sid, "role": "support"})
        claim = {"task_id": tid, "decision": decision, "mechanism": "cascade",
                 "entities": _entities_from(task_text), "window": _window_from(task_text),
                 "confidence": 0.7, "evidence": evidence[:2],
                 "narrative": f"mock policy {self.policy}"}
        return LLMReply(text=json.dumps(claim), input_tokens=300, output_tokens=80)

    def _decide(self, tid: str, messages: list[dict[str, Any]], tools: list[ToolSpec]) -> str:
        if self.policy == "always_present":
            return "present"
        if self.policy == "always_absent":
            return "absent"
        if self.policy == "coin_flip":
            import random
            return "present" if random.Random(f"{self.seed}:{tid}").random() < 0.5 else "absent"
        if self.policy == "static_oracle":
            # answers from order-blind evidence only: the static surface is identical across a
            # matched pair, so this predictor is forced to give the same answer to both worlds
            return "present"
        for m in reversed(messages):
            if m.get("role") == "tool" and '"reached": true' in m["content"].lower():
                return "present"
            if m.get("role") == "tool" and '"reached": false' in m["content"].lower():
                return "absent"
        return "insufficient"


def _task_id_from(text: str) -> str:
    for tok in text.split():
        if tok.startswith("t_"):
            return tok.rstrip(".")
    return "t_unknown"


def _entities_from(text: str) -> list[int]:
    import re
    m = re.findall(r"node (\d+)|account (\d+)", text)
    out = [int(a or b) for a, b in m]
    return out[:4] or [0]


def _window_from(text: str) -> list[int]:
    import re
    m = re.search(r"\[(\d+), (\d+)\]", text)
    return [int(m.group(1)), int(m.group(2))] if m else [0, 1]
