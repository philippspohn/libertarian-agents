"""OpenAI Responses API.

Stateless (`store=False`) with `include=["reasoning.encrypted_content"]`, so
reasoning survives across turns while we keep control of the item list.

Compaction is server-side: `context_management` makes the API compact in
stream once the rendered context crosses the threshold, and the response
carries back opaque encrypted `compaction` items. Those items subsume
everything before them -- verified: dropping the original input and passing
only the compaction items forward still answers questions about the dropped
content. That is strictly better than summarizing to text, because reasoning
crosses the boundary encrypted rather than being flattened into prose.

If a model rejects `context_management`, the parameter is dropped and the
runner falls back to summarize-and-rebuild. Note that encrypted state is bound
to the model that produced it: swapping models forces a manual compaction, it
does not carry over.
"""

from __future__ import annotations

import re
from typing import Any

from openai import BadRequestError

from ..llm import openai_client, usage_from_response
from ..models import RunnerConfig, UsageRow
from ..tools.base import ToolSpec
from .base import Provider, ToolCall, Turn, parse_arguments

# Params we will drop one at a time if the model rejects them, so an unusual
# model does not take the whole runner down.
OPTIONAL_PARAMS = ["context_management", "verbosity", "reasoning", "include", "temperature", "text"]


def tool_payload(spec: ToolSpec) -> dict[str, Any]:
    if spec.name == "web_search":
        return {"type": "web_search"}
    return {
        "type": "function",
        "name": spec.name,
        "description": spec.description,
        "parameters": spec.parameters,
    }


class OpenAIProvider(Provider):
    name = "openai"
    native_compaction = True

    COMPACT_THRESHOLD_MIN = 1000
    """The API's floor for compact_threshold."""

    def __init__(self, config: RunnerConfig):
        self.config = config
        self.model = config.model
        self.client = openai_client()
        self._disabled: set[str] = set()

    # ------------------------------------------------------------------ api

    def _kwargs(
        self, instructions: str, items: list[dict], tools: list[ToolSpec],
        max_output_tokens: int | None = None,
    ) -> dict:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "instructions": instructions,
            "input": items,
            "store": False,
        }
        if tools:
            # web_search is hosted by OpenAI inside this same Responses
            # request. No delegated helper-model call is involved.
            kwargs["tools"] = [tool_payload(t) for t in tools]
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = max(1, max_output_tokens)
        if "text" not in self._disabled:
            text: dict[str, Any] = {"format": {"type": "text"}}
            if self.config.verbosity and "verbosity" not in self._disabled:
                text["verbosity"] = self.config.verbosity
            kwargs["text"] = text
        if self.config.reasoning_effort and "reasoning" not in self._disabled:
            kwargs["reasoning"] = {"effort": self.config.reasoning_effort, "summary": "auto"}
        if "include" not in self._disabled:
            kwargs["include"] = ["reasoning.encrypted_content"]
        if self.config.temperature is not None and "temperature" not in self._disabled:
            kwargs["temperature"] = self.config.temperature
        if self.uses_native_compaction:
            kwargs["context_management"] = [
                {
                    "type": "compaction",
                    "compact_threshold": max(
                        self.COMPACT_THRESHOLD_MIN, self.config.compact_at_input_tokens
                    ),
                }
            ]
        return kwargs

    @property
    def uses_native_compaction(self) -> bool:
        """False once the model has rejected the parameter, so the runner can
        take over with manual compaction."""
        return "context_management" not in self._disabled

    def _create(self, kwargs: dict):
        for _ in range(len(OPTIONAL_PARAMS) + 1):
            try:
                return self.client.responses.create(**kwargs)
            except BadRequestError as exc:
                dropped = self._drop_offending_param(kwargs, str(exc))
                if not dropped:
                    raise
        raise RuntimeError("exhausted parameter fallbacks")

    def _drop_offending_param(self, kwargs: dict, message: str) -> bool:
        for param in OPTIONAL_PARAMS:
            if param in self._disabled:
                continue
            if re.search(rf"\b{param}\b", message) and (param in kwargs or param == "verbosity"):
                self._disabled.add(param)
                kwargs.pop(param, None)
                if param == "verbosity" and isinstance(kwargs.get("text"), dict):
                    kwargs["text"].pop("verbosity", None)
                return True
        return False

    # -------------------------------------------------------------- protocol

    def generate(
        self, *, instructions: str, items: list[dict], tools: list[ToolSpec],
        max_output_tokens: int | None = None,
    ) -> Turn:
        resp = self._create(self._kwargs(instructions, items, tools, max_output_tokens))
        turn = Turn(usage=usage_from_response(self.model, resp.usage))
        for item in resp.output or []:
            data = item.model_dump(exclude_none=True) if hasattr(item, "model_dump") else dict(item)
            turn.items.append(data)
            if data.get("type") == "function_call":
                args, err = parse_arguments(data.get("arguments", "{}"))
                turn.tool_calls.append(
                    ToolCall(
                        call_id=data.get("call_id") or data.get("id", ""),
                        name=data.get("name", ""),
                        arguments=args,
                        raw_arguments=data.get("arguments", ""),
                        parse_error=err,
                    )
                )
            elif data.get("type") == "compaction":
                turn.compaction_items.append(data)
            elif data.get("type") == "reasoning":
                parts = [s.get("text", "") for s in data.get("summary", []) if isinstance(s, dict)]
                turn.reasoning = "\n".join(p for p in parts if p)
            elif data.get("type") == "web_search_call":
                turn.hosted_tools.append("web_search")
        turn.text = (resp.output_text or "").strip()
        return turn

    def user_item(self, text: str) -> dict:
        return {"role": "user", "content": [{"type": "input_text", "text": text}]}

    def tool_result_item(self, call: ToolCall, output: str) -> dict:
        return {"type": "function_call_output", "call_id": call.call_id, "output": output}

    def summarize(
        self, *, instructions: str, items: list[dict], prompt: str,
        max_output_tokens: int | None = None,
    ) -> tuple[str, UsageRow]:
        kwargs = self._kwargs(
            instructions, items + [self.user_item(prompt)], [], max_output_tokens
        )
        kwargs.pop("include", None)
        # A one-off summary of the full context: do not let the server compact
        # the very thing we are asking the model to read.
        kwargs.pop("context_management", None)
        if isinstance(kwargs.get("text"), dict):
            kwargs["text"]["verbosity"] = "medium"
        resp = self._create(kwargs)
        return (resp.output_text or "").strip(), usage_from_response(self.model, resp.usage)
