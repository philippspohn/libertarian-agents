"""Configuration models.

The split that matters:

  * Anything in here that lives in the CONTROL PLANE (`RunnerConfig`,
    `EnvConfig`) is stored host-side in `control.db`, outside the sandbox.
    Agents cannot read or write it. Budgets in particular are enforced from
    token counts the orchestrator reads off API responses it made itself --
    never from anything inside the environment.
  * Anything an agent may change about its own behaviour (its instructions,
    its memory) lives as a plain file inside the environment.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

RunnerState = Literal["inactive", "active", "waiting", "finished"]

DEFAULT_TOOLS = [
    "shell",
    "read_file",
    "read_summary",
    "write_file",
    "edit_file",
    "delete_file",
    "web_search",
    "list_agents",
    "set_status",
    "send_message",
    "check_inbox",
    "read_history",
    "join_channel",
    "sleep",
    "finish",
]


class Budgets(BaseModel):
    """Hard caps. Enforced host-side; exceeding them stops the runner."""

    input_tokens: int = 1_000_000
    output_tokens: int = 100_000


class RunnerConfig(BaseModel):
    provider: Literal["openai", "openrouter"] = "openai"
    model: str = "gpt-5.6-luna"
    reasoning_effort: Optional[str] = "low"
    verbosity: Optional[str] = "low"
    temperature: Optional[float] = None

    tools: list[str] = Field(default_factory=lambda: list(DEFAULT_TOOLS))
    budgets: Budgets = Field(default_factory=Budgets)

    # Context management
    compact_at_input_tokens: int = 100_000
    native_compaction: bool = True
    """Use the provider's server-side compaction where it has one (OpenAI
    Responses `context_management`). It carries encrypted reasoning across the
    boundary, which a text summary cannot. Falls back to summarize-and-rebuild
    when the provider or model does not support it."""
    memory_char_limit: int = 6000
    memoryless: bool = False
    """Reset the whole conversation every time the agent sleeps. For cheap
    subagent-style profiles that should not carry context between wakes."""

    # Tool-output compression
    shell_head_lines: int = 3
    shell_tail_lines: int = 3
    shell_line_chars: int = 400
    shell_timeout: int = 120

    max_steps_per_wake: int = 200
    extra: dict[str, Any] = Field(default_factory=dict)


class EnvConfig(BaseModel):
    sandbox: Literal["docker", "local"] = "docker"
    image: str = "python:3.12-slim"
    goal: str = ""
    """The project goal. Injected as the user turn for every agent and
    mirrored to `<root>/shared/GOAL.md`."""

    max_message_chars: int = 4000
    summary_model: str = "gpt-5.6-luna"
    secrets: list[str] = Field(default_factory=list)
    """Names of host env vars to forward into the sandbox."""

    input_token_cap: Optional[int] = None
    """Environment-wide kill switch across all runners."""


class UsageRow(BaseModel):
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: "UsageRow") -> "UsageRow":
        return UsageRow(
            input_tokens=self.input_tokens + other.input_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )
