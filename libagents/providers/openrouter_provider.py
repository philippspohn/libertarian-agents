"""OpenRouter via the chat-completions shape.

No server-side state and no encrypted reasoning, so compaction here is purely
manual: ask the model to summarize itself, then rebuild. The one thing to be
careful about is the truncation seam -- an assistant message carrying
tool_calls must always keep its matching `role: "tool"` replies, or the API
rejects the request. `trim_to_valid_prefix` enforces that.
"""

from __future__ import annotations

from typing import Any

from ..llm import openrouter_client, usage_from_response
from ..models import RunnerConfig, UsageRow
from ..tools.base import ToolSpec
from .base import Provider, ToolCall, Turn, parse_arguments


def tool_payload(spec: ToolSpec) -> dict[str, Any]:
    if spec.name == "web_search":
        return {
            "type": "openrouter:web_search",
            "parameters": {
                "engine": "auto",
                "max_results": 5,
                "max_total_results": 10,
            },
        }
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    }


def trim_to_valid_prefix(items: list[dict]) -> list[dict]:
    """Drop trailing tool calls whose results are missing, and leading tool
    results whose call is missing."""
    out = list(items)
    while out and out[0].get("role") == "tool":
        out.pop(0)
    pending: set[str] = set()
    for item in out:
        if item.get("role") == "assistant":
            pending |= {tc["id"] for tc in item.get("tool_calls") or []}
        elif item.get("role") == "tool":
            pending.discard(item.get("tool_call_id"))
    while pending and out:
        last = out.pop()
        if last.get("role") == "assistant":
            pending -= {tc["id"] for tc in last.get("tool_calls") or []}
    return out


class OpenRouterProvider(Provider):
    name = "openrouter"
    native_compaction = False

    def __init__(self, config: RunnerConfig):
        self.config = config
        self.model = config.model
        self.client = openrouter_client()

    def _messages(self, instructions: str, items: list[dict]) -> list[dict]:
        return [{"role": "system", "content": instructions}] + trim_to_valid_prefix(items)

    def _call(
        self, instructions: str, items: list[dict], tools: list[ToolSpec],
        max_output_tokens: int | None = None,
    ):
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._messages(instructions, items),
        }
        if tools:
            kwargs["tools"] = [tool_payload(t) for t in tools]
            kwargs["tool_choice"] = "auto"
        if max_output_tokens is not None:
            kwargs["max_tokens"] = max(1, max_output_tokens)
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        extra: dict[str, Any] = {}
        if self.config.reasoning_effort:
            extra["reasoning"] = {"effort": self.config.reasoning_effort}
        if any(t.name == "web_search" for t in tools):
            extra["max_tool_calls"] = 3
        if extra:
            kwargs["extra_body"] = extra
        return self.client.chat.completions.create(**kwargs)

    def generate(
        self, *, instructions: str, items: list[dict], tools: list[ToolSpec],
        max_output_tokens: int | None = None,
    ) -> Turn:
        resp = self._call(instructions, items, tools, max_output_tokens)
        choice = resp.choices[0]
        msg = choice.message
        turn = Turn(usage=usage_from_response(self.model, resp.usage))

        assistant: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
            for tc in msg.tool_calls:
                args, err = parse_arguments(tc.function.arguments)
                turn.tool_calls.append(
                    ToolCall(tc.id, tc.function.name, args, tc.function.arguments, err)
                )
        turn.items.append(assistant)
        turn.text = (msg.content or "").strip()
        turn.reasoning = (getattr(msg, "reasoning", None) or "") if hasattr(msg, "reasoning") else ""
        server_use = getattr(resp.usage, "server_tool_use", None)
        searches = (
            server_use.get("web_search_requests", 0)
            if isinstance(server_use, dict)
            else getattr(server_use, "web_search_requests", 0)
        ) or 0
        turn.hosted_tools.extend(["web_search"] * searches)
        return turn

    def user_item(self, text: str) -> dict:
        return {"role": "user", "content": text}

    def tool_result_item(self, call: ToolCall, output: str) -> dict:
        return {"role": "tool", "tool_call_id": call.call_id, "content": output}

    def summarize(
        self, *, instructions: str, items: list[dict], prompt: str,
        max_output_tokens: int | None = None,
    ) -> tuple[str, UsageRow]:
        resp = self._call(
            instructions, items + [self.user_item(prompt)], [], max_output_tokens
        )
        return (resp.choices[0].message.content or "").strip(), usage_from_response(self.model, resp.usage)
