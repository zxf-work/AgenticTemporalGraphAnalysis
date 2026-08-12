"""Token accounting and a hard spend ceiling.

The runner checks the ceiling *before* dispatching each episode and stops the sweep when the next
run could cross it. Cost is computed from the tokens the providers actually report, accumulated
across the whole run directory, so a resumed sweep counts what earlier attempts already spent
rather than starting from zero.

Prices are per million tokens, verified 2026-08-11.
"""

from __future__ import annotations

import json
import pathlib
import threading
from dataclasses import dataclass

#: USD per million tokens, (input, output)
PRICES: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-opus-5": (5.00, 25.00),
    "claude-fable-5": (10.00, 50.00),
    "claude-sonnet-5": (2.00, 10.00),      # introductory rate through 2026-08-31 ($3/$15 after)
    "claude-haiku-4-5": (1.00, 5.00),
    # OpenAI (billed to a separate account from the Anthropic budget)
    "gpt-5": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
    # deterministic reference algorithm: no API calls
    "script-a1-bisect": (0.0, 0.0),
}

#: providers whose spend counts against the capped budget
CAPPED_PREFIXES = ("claude-",)


def price_of(model: str) -> tuple[float, float]:
    if model in PRICES:
        return PRICES[model]
    for known, p in PRICES.items():                # tolerate dated suffixes
        if model.startswith(known):
            return p
    return (0.0, 0.0)


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = price_of(model)
    return input_tokens / 1e6 * pin + output_tokens / 1e6 * pout


def is_capped(model: str) -> bool:
    return any(model.startswith(p) for p in CAPPED_PREFIXES)


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Spend:
    capped_usd: float = 0.0
    uncapped_usd: float = 0.0
    episodes: int = 0


class BudgetGuard:
    """Thread-safe running total with a hard ceiling on capped providers.

    ``reserve()`` is called before an episode starts and raises once the ceiling is reached, so a
    sweep degrades to "stops early with everything so far recorded" rather than "silently spends
    past the cap". Estimated per-episode cost seeds the reservation; the actual token counts
    replace it when the episode finishes.
    """

    def __init__(self, limit_usd: float | None, *, estimate_usd: float = 0.50):
        self.limit = limit_usd
        self.estimate = estimate_usd
        self.spend = Spend()
        self._lock = threading.Lock()
        self._stopped = False

    @classmethod
    def from_run_dir(cls, out: pathlib.Path, limit_usd: float | None,
                     estimate_usd: float = 0.50) -> "BudgetGuard":
        """Seed from results already on disk so a resumed sweep does not double its budget."""
        g = cls(limit_usd, estimate_usd=estimate_usd)
        results = pathlib.Path(out) / "results.jsonl"
        if results.exists():
            seen: dict[str, dict] = {}
            for line in results.read_text().splitlines():
                if line.strip():
                    d = json.loads(line)
                    seen[d.get("run_id", line)] = d
            for d in seen.values():
                g.record(d.get("model", ""), int(d.get("input_tokens", 0)),
                         int(d.get("output_tokens", 0)))
        return g

    def reserve(self, model: str) -> None:
        if self.limit is None or not is_capped(model):
            return
        with self._lock:
            if self.spend.capped_usd + self.estimate > self.limit:
                self._stopped = True
                raise BudgetExceeded(
                    f"budget ceiling ${self.limit:.2f} reached "
                    f"(spent ${self.spend.capped_usd:.2f} over {self.spend.episodes} episodes)")

    def record(self, model: str, input_tokens: int, output_tokens: int) -> float:
        c = cost_usd(model, input_tokens, output_tokens)
        with self._lock:
            if is_capped(model):
                self.spend.capped_usd += c
            else:
                self.spend.uncapped_usd += c
            self.spend.episodes += 1
        return c

    @property
    def stopped(self) -> bool:
        return self._stopped

    def line(self) -> str:
        cap = f"/${self.limit:.0f}" if self.limit else ""
        return (f"spend: ${self.spend.capped_usd:.2f}{cap} capped, "
                f"${self.spend.uncapped_usd:.2f} uncapped, {self.spend.episodes} episodes")


def estimate_sweep(n_episodes: int, model: str, *, mean_in: int = 45000,
                   mean_out: int = 5000) -> float:
    """Rough forecast for planning a grid before running it."""
    return n_episodes * cost_usd(model, mean_in, mean_out)
