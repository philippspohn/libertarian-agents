"""OpenAI Responses API.

Stateless (`store=False`) with `include=["reasoning.encrypted_content"]`, so
reasoning survives across turns while we keep full control of the item list --
which is what makes compaction ours to define. Note that encrypted reasoning
is bound to the model that produced it: swapping models forces a compaction
(see `runner.compact`), it does not carry over.
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
OPTIONAL_PARAMS = ["verbosity", "reasoning", "include", "temperature", "text"]


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, config: RunnerConfig):
        self.config = config
        self.model = config.model
        self.client = openai_client()
        self._disabled: set[str] = set()

    # ------------------------------------------------------------------ api

    def _kwargs(self, instructions: str, items: list[dict], tools: list[ToolSpec]) -> dict:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "instructions": instructions,
            "input": items,
            "store": False,
        }
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
                for t in tools
            ]
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
        return kwargs

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

    def generate(self, *, instructions: str, items: list[dict], tools: list[ToolSpec]) -> Turn:
        resp = self._create(self._kwargs(instructions, items, tools))
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
            elif data.get("type") == "reasoning":
                parts = [s.get("text", "") for s in data.get("summary", []) if isinstance(s, dict)]
                turn.reasoning = "\n".join(p for p in parts if p)
        turn.text = (resp.output_text or "").strip()
        return turn

    def user_item(self, text: str) -> dict:
        return {"role": "user", "content": [{"type": "input_text", "text": text}]}

    def tool_result_item(self, call: ToolCall, output: str) -> dict:
        return {"type": "function_call_output", "call_id": call.call_id, "output": output}

    def summarize(self, *, instructions: str, items: list[dict], prompt: str) -> tuple[str, UsageRow]:
        kwargs = self._kwargs(instructions, items + [self.user_item(prompt)], [])
        kwargs.pop("include", None)
        if isinstance(kwargs.get("text"), dict):
            kwargs["text"]["verbosity"] = "medium"
        resp = self._create(kwargs)
        return (resp.output_text or "").strip(), usage_from_response(self.model, resp.usage)
