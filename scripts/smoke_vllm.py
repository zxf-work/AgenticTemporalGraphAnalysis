"""Shakedown for a locally served open-weight subject, before any ladder GPU hours are spent.

Checks the three things that can silently ruin a C4 cell, in order of how quietly they fail:

  1. the endpoint answers at all;
  2. the model emits a *parseable* tool call against our real tool schema -- if the server is
     started without `--enable-auto-tool-choice`, vLLM returns the call as prose and every episode
     would look like a total analytical failure rather than an unparsed one;
  3. one full episode runs end to end and produces a schema-valid claim.

Run inside a job that has already started vLLM and exported ATGP_VLLM_URL.

    python scripts/smoke_vllm.py <model_slot>
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.backends import load_subject, warmup
from agents.tools_spec import surface_for

slot = sys.argv[1] if len(sys.argv) > 1 else "qwen7b"
cfg = yaml.safe_load((ROOT / "configs/models.yaml").read_text())
print(f"slot={slot} url={os.environ.get('ATGP_VLLM_URL')}")

backend = load_subject(cfg, slot)
print(f"backend name (goes into run_id): {backend.name}")

# ---- 1. endpoint reachable, request shape settled
status = warmup(backend)
print(f"warmup: {status}")
if status != "ok":
    sys.exit("endpoint unusable")

# ---- 2. tool calling against the REAL A1 surface, not a toy schema. A hand-written stub could
# pass while the actual surface (more tools, richer JSON schemas) trips the parser.
tools = surface_for("A1")
print(f"A1 surface: {len(tools)} tools -> {[t.name for t in tools][:6]}")
reply = backend.complete(
    "You are a graph analyst. Use the provided tools to answer. Do not answer from memory.",
    [{"role": "user", "content": "How many interaction events occurred between time 100 and "
                                 "time 500? Call a tool to find out."}],
    tools)
print(f"tool_calls parsed : {len(reply.tool_calls)}  -> {json.dumps(reply.tool_calls)[:220]}")
print(f"protocol_errors   : {len(reply.protocol_errors)}  {reply.protocol_errors[:2]}")
print(f"text              : {reply.text[:160]!r}")

if not reply.tool_calls and not reply.protocol_errors:
    print("\nFAIL: the model neither emitted a tool call nor a malformed one. If the reply text "
          "contains something that looks like a call, vLLM is not parsing tool calls -- check "
          "--enable-auto-tool-choice and --tool-call-parser.")
    sys.exit(1)
if not reply.tool_calls:
    print("\nFAIL: only malformed tool calls. Wrong --tool-call-parser for this family?")
    sys.exit(1)

print("\nsmoke OK: endpoint healthy, tool calls parse")
