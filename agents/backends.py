"""Provider backends implementing the harness ``Backend`` protocol.

Both providers see the same tool surface, the same system prompt, the same budget, and the same
output contract; only the wire format differs. Provider-specific quirks are absorbed here so that
nothing about them leaks into the agent loop, which must stay identical across conditions for the
A1/A3/A4 comparison to mean anything.

The internal message format is provider-neutral:

    {"role": "user",      "content": str}
    {"role": "assistant", "content": str, "tool_calls": [{"id", "name", "args"}]}
    {"role": "tool",      "tool_call_id": str, "name": str, "content": str}
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agents.harness import LLMReply
from agents.tools_spec import ToolSpec


class BackendError(RuntimeError):
    pass


def _retry(fn, attempts: int, label: str):
    """Retry transport failures only. Content-level failures are never retried (plan §14)."""
    last: Exception | None = None
    for i in range(attempts + 1):
        try:
            return fn()
        except Exception as exc:                     # noqa: BLE001 - provider SDK exceptions vary
            last = exc
            msg = f"{type(exc).__name__}: {exc}"
            transient = any(k in msg.lower() for k in
                            ("timeout", "connection", "overloaded", "rate", "429", "500", "502",
                             "503", "529", "temporarily"))
            if not transient or i == attempts:
                raise BackendError(f"{label}: {msg}") from exc
            time.sleep(min(2 ** i * 2, 30))
    raise BackendError(f"{label}: {last}")


# ======================================================================================
# Anthropic
# ======================================================================================

class AnthropicBackend:
    def __init__(self, model: str, *, temperature: float = 0.0, max_tokens: int = 4096,
                 retries: int = 1):
        import anthropic
        self.name = model
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        # Newer models reject `temperature` outright ("deprecated for this model"). Learn the
        # accepted shape on the first call and keep it for the rest of the matrix, so every run
        # of a given model is issued identically.
        self._send_temperature = temperature is not None

    def complete(self, system: str, messages: list[dict[str, Any]],
                 tools: list[ToolSpec]) -> LLMReply:
        def build() -> dict[str, Any]:
            payload: dict[str, Any] = {
                "model": self.model,
                "system": system,
                "max_tokens": self.max_tokens,
                "messages": _to_anthropic(messages),
            }
            if tools:
                payload["tools"] = [{"name": t.name, "description": t.description,
                                     "input_schema": t.parameters} for t in tools]
            if self._send_temperature:
                payload["temperature"] = self.temperature
            return payload

        try:
            resp = _retry(lambda: self._client.messages.create(**build()), self.retries,
                          f"anthropic/{self.model}")
        except BackendError as exc:
            if "temperature" in str(exc).lower() and self._send_temperature:
                self._send_temperature = False
                resp = _retry(lambda: self._client.messages.create(**build()), self.retries,
                              f"anthropic/{self.model}")
            else:
                raise

        text_parts, calls = [], []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append({"id": block.id, "name": block.name, "args": dict(block.input)})
        return LLMReply(text="\n".join(text_parts), tool_calls=calls,
                        input_tokens=resp.usage.input_tokens,
                        output_tokens=resp.usage.output_tokens)


def _to_anthropic(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Neutral messages -> Anthropic content blocks.

    Consecutive tool results must be merged into a single user turn: the API rejects a sequence of
    separate user messages carrying tool_result blocks.
    """
    out: list[dict[str, Any]] = []
    pending_results: list[dict[str, Any]] = []

    def flush() -> None:
        if pending_results:
            out.append({"role": "user", "content": list(pending_results)})
            pending_results.clear()

    for m in messages:
        role = m["role"]
        if role == "tool":
            pending_results.append({"type": "tool_result",
                                    "tool_use_id": m.get("tool_call_id") or m.get("name", ""),
                                    "content": str(m["content"])})
            continue
        flush()
        if role == "assistant":
            blocks: list[dict[str, Any]] = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for c in m.get("tool_calls") or []:
                blocks.append({"type": "tool_use", "id": c["id"], "name": c["name"],
                               "input": c.get("args") or {}})
            out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "."}]})
        else:
            out.append({"role": "user", "content": str(m["content"])})
    flush()
    return out


# ======================================================================================
# OpenAI
# ======================================================================================

class OpenAIBackend:
    def __init__(self, model: str, *, temperature: float = 0.0, max_tokens: int = 4096,
                 retries: int = 1, base_url: str | None = None, api_key_env: str = "OPENAI_API_KEY",
                 label: str | None = None):
        import openai
        self.name = label or model
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        # base_url lets the same OpenAI wire protocol address a locally served open-weight model
        # (vLLM exposes an OpenAI-compatible endpoint), so the agent loop is byte-identical across
        # hosted and local subjects -- the comparison is between models, not between harnesses.
        kw: dict[str, Any] = {"api_key": os.environ.get(api_key_env) or "EMPTY"}
        if base_url:
            kw["base_url"] = base_url
            kw["timeout"] = 900.0
        self._client = openai.OpenAI(**kw)
        # discovered on first call and remembered, so the whole matrix uses one shape
        self._token_param: str | None = None
        if base_url:
            # A local vLLM server accepts `max_tokens` and, when it is omitted, defaults it to the
            # entire remaining context -- then rejects the request because prompt + that default
            # exceeds max_model_len by a few tokens of chat-template overhead. Send it explicitly
            # from the first call. This also keeps the generation cap identical to the hosted
            # subjects, so output length is not a hidden difference between models.
            self._token_param = "max_tokens"
        self._send_temperature = temperature is not None

    def complete(self, system: str, messages: list[dict[str, Any]],
                 tools: list[ToolSpec]) -> LLMReply:
        msgs = [{"role": "system", "content": system}] + _to_openai(messages)
        base: dict[str, Any] = {"model": self.model, "messages": msgs}
        if tools:
            base["tools"] = [{"type": "function",
                              "function": {"name": t.name, "description": t.description,
                                           "parameters": t.parameters}} for t in tools]

        def call() -> Any:
            payload = dict(base)
            if self._token_param:
                payload[self._token_param] = self.max_tokens
            if self._send_temperature:
                payload["temperature"] = self.temperature
            return self._client.chat.completions.create(**payload)

        try:
            resp = _retry(call, self.retries, f"openai/{self.model}")
        except BackendError as exc:
            # GPT-5-class models reject `max_tokens` (want `max_completion_tokens`) and reject a
            # non-default temperature. Learn the accepted shape once, then retry.
            msg = str(exc).lower()
            changed = False
            if "max_tokens" in msg and self._token_param != "max_completion_tokens":
                self._token_param = "max_completion_tokens"
                changed = True
            if "temperature" in msg and self._send_temperature:
                self._send_temperature = False
                changed = True
            if not changed:
                raise
            resp = _retry(call, self.retries, f"openai/{self.model}")

        choice = resp.choices[0].message
        calls, protocol_errors = [], []
        offered = {t.name for t in tools}
        for c in (choice.tool_calls or []):
            raw = c.function.arguments or "{}"
            try:
                args = json.loads(raw)
            except json.JSONDecodeError:
                # Silently substituting {} here would charge a malformed emission to the agent's
                # analysis. Record it as a protocol failure and drop the call instead.
                protocol_errors.append(f"unparseable arguments for {c.function.name!r}: "
                                       f"{raw[:120]}")
                continue
            if not isinstance(args, dict):
                protocol_errors.append(f"non-object arguments for {c.function.name!r}: "
                                       f"{str(args)[:80]}")
                continue
            if offered and c.function.name not in offered:
                protocol_errors.append(f"called undeclared tool {c.function.name!r}")
                continue
            calls.append({"id": c.id, "name": c.function.name, "args": args})
        usage = resp.usage
        return LLMReply(text=choice.content or "", tool_calls=calls,
                        input_tokens=getattr(usage, "prompt_tokens", 0),
                        output_tokens=getattr(usage, "completion_tokens", 0),
                        protocol_errors=protocol_errors)


def _to_openai(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            out.append({"role": "assistant", "content": m.get("content") or None,
                        "tool_calls": [{"id": c["id"], "type": "function",
                                        "function": {"name": c["name"],
                                                     "arguments": json.dumps(c.get("args") or {})}}
                                       for c in m["tool_calls"]]})
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": m.get("tool_call_id", ""),
                        "content": str(m["content"])})
        else:
            out.append({"role": m["role"], "content": str(m["content"])})
    return out


# ======================================================================================
# factory
# ======================================================================================

def warmup(backend: Any) -> str:
    """Settle a backend's accepted request shape before any concurrent use.

    Both backends discover on their first call which parameters the model rejects (temperature on
    Opus 5, `max_tokens` on the GPT-5 family) and remember the answer. Under a thread pool that
    discovery would otherwise race: several workers would fail the same way at once and each pay a
    retry. One trivial call up front removes the race and the wasted requests.
    """
    try:
        backend.complete("Reply with the single word OK.",
                         [{"role": "user", "content": "OK"}], [])
        return "ok"
    except Exception as exc:                                  # noqa: BLE001
        return f"warmup failed: {type(exc).__name__}: {str(exc)[:160]}"


def make_backend(model_id: str, provider: str, *, temperature: float = 0.0,
                 max_tokens: int = 4096, retries: int = 1, base_url: str | None = None,
                 label: str | None = None):
    if provider == "anthropic":
        return AnthropicBackend(model_id, temperature=temperature, max_tokens=max_tokens,
                                retries=retries)
    if provider == "openai":
        return OpenAIBackend(model_id, temperature=temperature, max_tokens=max_tokens,
                             retries=retries)
    if provider == "vllm":
        # locally served open-weight subject; base_url comes from the config or the environment
        return OpenAIBackend(model_id, temperature=temperature, max_tokens=max_tokens,
                             retries=retries,
                             base_url=(base_url or os.environ.get("ATGP_VLLM_URL")
                                       or "http://127.0.0.1:8000/v1"),
                             api_key_env="ATGP_VLLM_KEY", label=label or model_id)
    if provider == "mock":
        from agents.harness import MockBackend
        return MockBackend(policy=model_id)
    raise ValueError(f"unknown provider {provider!r}")


def load_subject(cfg: dict[str, Any], which: str = "primary"):
    """Instantiate a subject model from configs/models.yaml."""
    m = cfg["models"][which]
    d = cfg["defaults"]
    if not m.get("id"):
        raise ValueError(f"model slot {which!r} has no id configured")
    return make_backend(m["id"], m["provider"], temperature=d.get("temperature", 0.0),
                        max_tokens=d.get("max_output_tokens", 4096),
                        retries=d.get("transport_retries", 1),
                        base_url=m.get("base_url"), label=m.get("label"))
