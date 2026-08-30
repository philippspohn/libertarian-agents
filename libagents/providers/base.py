"""Provider abstraction.

Each provider keeps its own NATIVE item format in the conversation history --
we never normalise into a lowest-common-denominator shape, because that is
exactly how encrypted reasoning blocks and tool-call pairings get destroyed.
The runner only ever asks a provider for: give me a turn, wrap this text as a
user item, wrap this tool output as a result item.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..models import UsageRow
from ..tools.base import ToolSpec


@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""
    parse_error: str | None = None


@dataclass
class Turn:
    items: list[dict] = field(default_factory=list)
    compaction_items: list[dict] = field(default_factory=list)
    """Server-side compaction checkpoints emitted during this turn. Opaque and
    encrypted; everything before the first one can be dropped from the local
    item list because these carry it forward."""
    tool_calls: list[ToolCall] = field(default_factory=list)
    text: str = ""
    reasoning: str = ""
    usage: UsageRow = field(default_factory=UsageRow)


def parse_arguments(raw: str) -> tuple[dict, str | None]:
    try:
        parsed = json.loads(raw or "{}")
        if not isinstance(parsed, dict):
            return {}, "arguments must be a JSON object"
        return parsed, None
    except json.JSONDecodeError as exc:
        return {}, f"could not parse arguments as JSON: {exc}"


class Provider(Protocol):
    name: str
    model: str
    native_compaction: bool
    """Whether the provider compacts server-side. If not, the runner falls
    back to asking the model to summarize its own context."""

    def generate(self, *, instructions: str, items: list[dict], tools: list[ToolSpec]) -> Turn: ...
    def user_item(self, text: str) -> dict: ...
    def tool_result_item(self, call: ToolCall, output: str) -> dict: ...
    def summarize(self, *, instructions: str, items: list[dict], prompt: str) -> tuple[str, UsageRow]: ...
