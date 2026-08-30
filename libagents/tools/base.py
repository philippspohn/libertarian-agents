"""Tool framework.

Every tool returns a short `summary` for the model plus, optionally, the full
output -- which is always written to a file under the agent's `outputs/` dir
so nothing is actually lost. The model can then `read_file` (verbatim) or
`read_summary` (cheap-model digest) to go deeper. That spill-and-compress
policy lives here, once, rather than in each tool.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..board import Board
from ..events import EventLog
from ..models import EnvConfig, RunnerConfig, UsageRow
from ..paths import PathMapper


class Sleep(Exception):
    """Raised by the `sleep` tool: end inference, keep the runner wakeable."""

    def __init__(self, seconds: Optional[float], status: str = ""):
        super().__init__("sleep")
        self.seconds = seconds
        self.status = status


class Finish(Exception):
    """Raised by the `finish` tool: end inference for good."""

    def __init__(self, summary: str = ""):
        super().__init__("finish")
        self.summary = summary


class ToolError(Exception):
    """Recoverable: surfaced to the model as the tool's output."""


@dataclass
class ToolResult:
    summary: str
    full: Optional[str] = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentContext:
    env: str
    profile: str
    config: RunnerConfig
    env_config: EnvConfig
    sandbox: Any
    board: Board
    mapper: PathMapper
    events: EventLog
    profile_dir: Path
    used: UsageRow

    def status_line(self) -> str:
        b = self.config.budgets
        return (
            f"STATUS tokens_in={self.used.input_tokens}/{b.input_tokens} "
            f"tokens_out={self.used.output_tokens}/{b.output_tokens} "
            f"unread={self.board.unread_count(self.profile)}"
        )

    @property
    def outputs_dir(self) -> Path:
        d = self.profile_dir / "outputs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def spill(self, text: str, prefix: str = "out") -> str:
        """Write full output to a file and return its agent-facing path."""
        fp = self.outputs_dir / f"{prefix}-{uuid.uuid4().hex[:8]}.txt"
        fp.write_text(text, encoding="utf-8")
        return self.mapper.to_env(fp)

    def add_usage(self, model: str, usage: UsageRow) -> None:
        """Fold in tokens spent by helper calls (read_summary, web_search) so
        they count against the same budget as the agent's own turns."""
        from .. import control

        self.used.input_tokens += usage.input_tokens
        self.used.cached_input_tokens += usage.cached_input_tokens
        self.used.output_tokens += usage.output_tokens
        self.used.reasoning_tokens += usage.reasoning_tokens
        self.used.cost_usd += usage.cost_usd
        control.record_usage(self.env, self.profile, model, usage)

    def cwd(self) -> str:
        """Sandbox-side working directory for shell commands."""
        return self.mapper.to_env(self.profile_dir)


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict
    fn: Callable[[AgentContext, dict], ToolResult]
    ends_turn: bool = False


REGISTRY: dict[str, ToolSpec] = {}


def tool(name: str, description: str, parameters: dict, ends_turn: bool = False):
    def deco(fn: Callable[[AgentContext, dict], ToolResult]) -> Callable:
        REGISTRY[name] = ToolSpec(name, description.strip(), parameters, fn, ends_turn)
        return fn

    return deco


def obj(properties: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def compress(text: str, head: int, tail: int, line_chars: int) -> tuple[str, bool]:
    """First `head` and last `tail` lines, each clipped. Returns (text, elided)."""
    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    if not lines:
        return "", False

    def clip(ln: str) -> str:
        return ln if len(ln) <= line_chars else ln[:line_chars] + " ...[clipped]"

    if len(lines) <= head + tail:
        return "\n".join(clip(ln) for ln in lines), False
    shown = [clip(ln) for ln in lines[:head]]
    shown.append(f"... [{len(lines) - head - tail} more lines] ...")
    shown += [clip(ln) for ln in lines[-tail:]]
    return "\n".join(shown), True


def load_all() -> None:
    """Import every tool module so the registry is populated."""
    from . import control, files, messaging, shell, web  # noqa: F401


def specs_for(names: list[str]) -> list[ToolSpec]:
    load_all()
    missing = [n for n in names if n not in REGISTRY]
    if missing:
        raise KeyError(f"unknown tools: {', '.join(missing)}")
    return [REGISTRY[n] for n in names]
