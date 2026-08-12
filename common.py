"""Shared validation and hashing utilities.

Every pinned interface (task, claim, trace, invariants) validates through these helpers so that
rejection messages are uniform and machine-greppable.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Sequence


class SchemaError(ValueError):
    """Raised when an object violates a pinned interface."""


# --------------------------------------------------------------------------------------
# canonical serialisation + hashing
# --------------------------------------------------------------------------------------

def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no insignificant whitespace, stable float repr."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      default=_json_default)


def _json_default(o: Any) -> Any:
    if hasattr(o, "to_dict"):
        return o.to_dict()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def stable_hash(obj: Any, length: int = 16) -> str:
    """Content hash of any JSON-able object. Stable across processes and machines."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:length]


def digest_text(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


# --------------------------------------------------------------------------------------
# validators
# --------------------------------------------------------------------------------------

def require_keys(d: dict, keys: Iterable[str], where: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise SchemaError(f"{where}: missing required key(s) {missing}")


def forbid_keys(d: dict, keys: Iterable[str], where: str) -> None:
    """Used by the leakage audit: served artifacts must not carry generator metadata."""
    present = [k for k in keys if k in d]
    if present:
        raise SchemaError(f"{where}: forbidden key(s) present {present}")


def check_type(value: Any, types: type | tuple[type, ...], where: str) -> None:
    if isinstance(value, bool) and types is not bool and bool not in _as_tuple(types):
        raise SchemaError(f"{where}: bool is not an accepted type here")
    if not isinstance(value, types):
        names = "/".join(t.__name__ for t in _as_tuple(types))
        raise SchemaError(f"{where}: expected {names}, got {type(value).__name__}")


def _as_tuple(types: type | tuple[type, ...]) -> tuple[type, ...]:
    return types if isinstance(types, tuple) else (types,)


def check_enum(value: Any, allowed: Sequence[Any], where: str) -> None:
    if value not in allowed:
        raise SchemaError(f"{where}: {value!r} not in {list(allowed)}")


def check_interval(value: Any, where: str) -> tuple[int, int]:
    """A time interval is [start, end] with integer bounds and start <= end."""
    check_type(value, (list, tuple), where)
    if len(value) != 2:
        raise SchemaError(f"{where}: interval must have exactly 2 elements, got {len(value)}")
    lo, hi = value
    check_type(lo, int, f"{where}[0]")
    check_type(hi, int, f"{where}[1]")
    if lo > hi:
        raise SchemaError(f"{where}: start {lo} > end {hi}")
    return int(lo), int(hi)


def check_unit(value: Any, where: str) -> float:
    check_type(value, (int, float), where)
    if not 0.0 <= float(value) <= 1.0:
        raise SchemaError(f"{where}: {value} outside [0, 1]")
    return float(value)


def check_int_list(value: Any, where: str) -> list[int]:
    check_type(value, (list, tuple), where)
    out = []
    for i, v in enumerate(value):
        check_type(v, int, f"{where}[{i}]")
        out.append(int(v))
    return out
