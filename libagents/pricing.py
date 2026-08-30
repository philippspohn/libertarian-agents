"""Fallback price estimates in USD per 1M tokens.

Provider-reported cost is preferred. A missing model is unknown, not free.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_PRICING: dict[str, dict[str, float]] = {}


def _load() -> dict[str, dict[str, float]]:
    table = dict(DEFAULT_PRICING)
    override = os.environ.get("LIBAGENTS_PRICING")
    for candidate in filter(None, [override, str(Path.cwd() / "pricing.json")]):
        p = Path(candidate)
        if p.exists():
            table.update(json.loads(p.read_text()))
    return table


def cost(model: str, input_tokens: int, cached: int, output_tokens: int) -> float | None:
    table = _load()
    entry = None
    for key in sorted(table, key=len, reverse=True):
        if model.startswith(key):
            entry = table[key]
            break
    if not entry:
        return None
    fresh = max(0, input_tokens - cached)
    return (
        fresh * entry.get("input", 0.0)
        + cached * entry.get("cached_input", entry.get("input", 0.0))
        + output_tokens * entry.get("output", 0.0)
    ) / 1_000_000
