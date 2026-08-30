"""Fallback price estimates in USD per 1M tokens.

Provider-reported cost is preferred. A missing model is unknown, not free.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_PRICING: dict[str, dict[str, float]] = {
    # GPT-5.6 requests above 272K input tokens use the long-context rates for
    # the entire request. Cache writes are distinct from ordinary uncached
    # input and are reported by the Responses API on this model family.
    "gpt-5.6-sol": {
        "input": 4.00, "cached_input": 0.40, "cache_write": 5.00, "output": 20.00,
        "long_context_threshold": 272_000,
        "long_input": 8.00, "long_cached_input": 0.80,
        "long_cache_write": 10.00, "long_output": 30.00,
    },
    "gpt-5.6-terra": {
        "input": 2.00, "cached_input": 0.20, "cache_write": 2.50, "output": 12.00,
        "long_context_threshold": 272_000,
        "long_input": 4.00, "long_cached_input": 0.40,
        "long_cache_write": 5.00, "long_output": 18.00,
    },
    "gpt-5.6-luna": {
        "input": 0.20, "cached_input": 0.02, "cache_write": 0.25, "output": 1.20,
        "long_context_threshold": 272_000,
        "long_input": 0.40, "long_cached_input": 0.04,
        "long_cache_write": 0.50, "long_output": 1.80,
    },
    # OpenAI documents the unsuffixed alias as GPT-5.6 Sol.
    "gpt-5.6": {
        "input": 4.00, "cached_input": 0.40, "cache_write": 5.00, "output": 20.00,
        "long_context_threshold": 272_000,
        "long_input": 8.00, "long_cached_input": 0.80,
        "long_cache_write": 10.00, "long_output": 30.00,
    },
}


def _load() -> dict[str, dict[str, float]]:
    table = dict(DEFAULT_PRICING)
    override = os.environ.get("LIBAGENTS_PRICING")
    for candidate in filter(None, [override, str(Path.cwd() / "pricing.json")]):
        p = Path(candidate)
        if p.exists():
            table.update(json.loads(p.read_text()))
    return table


def cost(
    model: str,
    input_tokens: int,
    cached: int,
    output_tokens: int,
    cache_writes: int = 0,
) -> float | None:
    table = _load()
    entry = None
    for key in sorted(table, key=len, reverse=True):
        if model.startswith(key):
            entry = table[key]
            break
    if not entry:
        return None
    long_context = input_tokens > entry.get("long_context_threshold", float("inf"))
    prefix = "long_" if long_context else ""
    input_rate = entry.get(f"{prefix}input", entry.get("input", 0.0))
    cached_rate = entry.get(
        f"{prefix}cached_input", entry.get("cached_input", input_rate)
    )
    write_rate = entry.get(
        f"{prefix}cache_write", entry.get("cache_write", input_rate)
    )
    output_rate = entry.get(f"{prefix}output", entry.get("output", 0.0))
    fresh = max(0, input_tokens - cached - cache_writes)
    return (
        fresh * input_rate
        + cached * cached_rate
        + cache_writes * write_rate
        + output_tokens * output_rate
    ) / 1_000_000
