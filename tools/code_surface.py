"""A2 — programmable interface: raw temporal events in a sandboxed Python environment.

Milestone C3. The objection A2 answers is specific: *A3 only wins because A1 is an artificially
weak alternative.* A2 removes that objection by giving the agent strictly more information than
A1 and full generality — the raw `(u, v, t)` events and a Python interpreter — while giving it no
temporal semantics at all. If the pattern survives against A2, it is a statement about semantics
rather than about tool count.

Position in the hierarchy:

    A1   windowed aggregates, no ordering primitive   -> must reconstruct order by composition
    A2   raw events + Python, no temporal operators   -> must synthesise the algorithm
    A3   semantic temporal operators                  -> must specify intent correctly

A2 therefore has *no reconstruction frontier by construction*: timestamps are handed over
directly. That is the point. The question is not whether A2 can see time — it can — but whether
writing a correct time-respecting algorithm under a call budget is cheaper or more reliable than
specifying a semantic operator, and what new failure modes appear (wrong algorithm, mis-bound
constraint, stale intermediate state).

Sandboxing
----------
The code is model-generated, so it runs in a separate process with `-I -S` (isolated, no site),
a wall-clock timeout, and address-space and file-size limits applied via `resource` in a
pre-exec hook. The child receives events on stdin and may not import anything outside a small
allow-list. This is a research harness on hardware we control, not a hostile-code sandbox, and
the limits are here to stop runaway generated loops from taking down a sweep -- an infinite loop
in one episode must not cost the other eleven their GPU node.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: modules the generated code may import. Deliberately small: everything needed to write a
#: time-respecting algorithm over a list of tuples, and nothing that touches the host.
ALLOWED_IMPORTS = ("math", "collections", "itertools", "heapq", "bisect", "functools", "statistics")

MAX_OUTPUT_CHARS = 4000
DEFAULT_TIMEOUT_S = 20
MAX_ADDRESS_SPACE = 2 << 30      # 2 GiB

_CHILD = r'''
import sys, json, io, contextlib, builtins

payload = json.loads(sys.stdin.read())
events = [tuple(e) for e in payload["events"]]
horizon = tuple(payload["horizon"])
focus = payload["focus"]
code = payload["code"]

ALLOWED = set(payload["allowed_imports"])
_real_import = builtins.__import__
def _guarded_import(name, *a, **k):
    root = name.split(".")[0]
    if root not in ALLOWED:
        raise ImportError(
            f"import of {name!r} is not permitted in this environment; "
            f"available: {sorted(ALLOWED)}")
    return _real_import(name, *a, **k)
builtins.__import__ = _guarded_import

# `events` is the entire observable graph: a list of (u, v, t) tuples, already sorted by t.
env = {"events": events, "horizon": horizon, "focus": focus, "__name__": "__main__"}

buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        exec(compile(code, "<agent_code>", "exec"), env)
    out = {"ok": True, "stdout": buf.getvalue(), "error": ""}
except BaseException as exc:
    out = {"ok": False, "stdout": buf.getvalue(),
           "error": f"{type(exc).__name__}: {exc}"}
sys.stdout.write("\x00RESULT\x00" + json.dumps(out))
'''


def _limits() -> None:                                    # pragma: no cover - child process only
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (MAX_ADDRESS_SPACE, MAX_ADDRESS_SPACE))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))


def run_python(world: Any, code: str, *, focus: dict[str, Any] | None = None,
               timeout_s: int = DEFAULT_TIMEOUT_S) -> dict[str, Any]:
    """Execute agent-written code against the world's raw events.

    Returns the same shape whether the code succeeded, raised, or timed out, so a failure is data
    the agent can act on (and that we can count as a repair) rather than a harness error.
    """
    ev = world.events_in(int(world.horizon[0]), int(world.horizon[1]))
    payload = {
        "events": [[int(u), int(v), int(t)] for u, v, t in ev],
        "horizon": [int(world.horizon[0]), int(world.horizon[1])],
        "focus": focus or {},
        "code": code,
        "allowed_imports": list(ALLOWED_IMPORTS),
    }
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-S", "-c", _CHILD],
            input=json.dumps(payload), capture_output=True, text=True,
            timeout=timeout_s, preexec_fn=_limits,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "stdout": "", "n_events": len(payload["events"]),
                "error": f"execution exceeded {timeout_s}s and was terminated",
                "elapsed_ms": int((time.time() - t0) * 1000)}

    raw = proc.stdout or ""
    marker = raw.rfind("\x00RESULT\x00")
    if marker < 0:
        return {"ok": False, "stdout": raw[-MAX_OUTPUT_CHARS:], "n_events": len(payload["events"]),
                "error": (proc.stderr or "child produced no result")[-600:],
                "elapsed_ms": int((time.time() - t0) * 1000)}
    res = json.loads(raw[marker + len("\x00RESULT\x00"):])

    stdout = res["stdout"]
    truncated = len(stdout) > MAX_OUTPUT_CHARS
    if truncated:
        # Keep both ends: a long print loop usually carries its conclusion at the end, while the
        # start says what was computed. Silently dropping the tail would make a correct program
        # look like it produced nothing.
        head, tail = stdout[: MAX_OUTPUT_CHARS // 2], stdout[-MAX_OUTPUT_CHARS // 2:]
        stdout = f"{head}\n... [{len(res['stdout'])} chars total, middle elided] ...\n{tail}"
    return {"ok": bool(res["ok"]), "stdout": stdout, "error": res["error"],
            "n_events": len(payload["events"]), "output_truncated": truncated,
            "elapsed_ms": int((time.time() - t0) * 1000)}
