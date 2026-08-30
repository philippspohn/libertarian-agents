"""Shared LLM plumbing: clients, and the cheap-model helpers used by tools."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from openai import OpenAI

from .models import UsageRow
from .pricing import cost


@lru_cache(maxsize=4)
def openai_client(base_url: Optional[str] = None, api_key_env: str = "OPENAI_API_KEY") -> OpenAI:
    key = os.environ.get(api_key_env)
    if not key:
        raise RuntimeError(f"{api_key_env} is not set")
    return OpenAI(api_key=key, base_url=base_url)


def openrouter_client() -> OpenAI:
    return openai_client("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY")


def usage_from_response(model: str, usage) -> UsageRow:
    if usage is None:
        return UsageRow()
    input_details = (
        getattr(usage, "input_tokens_details", None)
        or getattr(usage, "prompt_tokens_details", None)
    )
    output_details = (
        getattr(usage, "output_tokens_details", None)
        or getattr(usage, "completion_tokens_details", None)
    )
    cached = getattr(input_details, "cached_tokens", 0) or 0
    reasoning = getattr(output_details, "reasoning_tokens", 0) or 0
    inp = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", 0) or 0
    out = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", 0) or 0
    reported_cost = getattr(usage, "cost", None)
    estimated_cost = None if reported_cost is not None else cost(model, inp, cached, out)
    return UsageRow(
        input_tokens=inp,
        cached_input_tokens=cached,
        output_tokens=out,
        reasoning_tokens=reasoning,
        cost_usd=float(reported_cost) if reported_cost is not None else (estimated_cost or 0.0),
        cost_known=reported_cost is not None or estimated_cost is not None,
    )


def quick_call(
    provider: str,
    model: str,
    prompt: str,
    *,
    max_chars: int = 200_000,
    max_output_tokens: int = 1500,
) -> tuple[str, UsageRow]:
    """One-shot, no-tools call on a cheap model. Used by read_summary so its
    output reaches the agent already compressed."""
    if provider == "openrouter":
        resp = openrouter_client().chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt[:max_chars]}],
            max_tokens=max_output_tokens,
        )
        return (
            (resp.choices[0].message.content or "").strip(),
            usage_from_response(model, resp.usage),
        )

    client = openai_client()
    resp = client.responses.create(
        model=model,
        input=[{"role": "user", "content": [{"type": "input_text", "text": prompt[:max_chars]}]}],
        text={"format": {"type": "text"}, "verbosity": "low"},
        reasoning={"effort": "low"},
        max_output_tokens=max_output_tokens,
        store=False,
    )
    return (resp.output_text or "").strip(), usage_from_response(model, resp.usage)
