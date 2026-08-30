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
    cached = getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0) or 0
    reasoning = getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", 0) or 0
    inp = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", 0) or 0
    out = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", 0) or 0
    return UsageRow(
        input_tokens=inp,
        cached_input_tokens=cached,
        output_tokens=out,
        reasoning_tokens=reasoning,
        cost_usd=cost(model, inp, cached, out),
    )


def quick_call(model: str, prompt: str, max_chars: int = 200_000) -> tuple[str, UsageRow]:
    """One-shot, no-tools call on a cheap model. Used by read_summary and
    web_search so their outputs reach the agent already compressed."""
    client = openai_client()
    resp = client.responses.create(
        model=model,
        input=[{"role": "user", "content": [{"type": "input_text", "text": prompt[:max_chars]}]}],
        text={"format": {"type": "text"}, "verbosity": "low"},
        reasoning={"effort": "low"},
        store=False,
    )
    return (resp.output_text or "").strip(), usage_from_response(model, resp.usage)


def web_search_call(model: str, query: str) -> tuple[str, UsageRow]:
    """Search via the provider's built-in web_search tool and return a digest."""
    client = openai_client()
    resp = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Search the web and answer concisely. Give the key facts in at "
                            "most 8 short bullet points, each with a source URL.\n\n"
                            f"Query: {query}"
                        ),
                    }
                ],
            }
        ],
        tools=[{"type": "web_search"}],
        text={"format": {"type": "text"}, "verbosity": "low"},
        reasoning={"effort": "low"},
        store=False,
    )
    return (resp.output_text or "").strip(), usage_from_response(model, resp.usage)
