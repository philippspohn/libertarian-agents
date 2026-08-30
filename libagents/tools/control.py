from __future__ import annotations

from .base import AgentContext, Finish, Sleep, ToolResult, obj, tool


@tool(
    "sleep",
    """
    Stop thinking and wait. This ends your current run: you keep your context
    but burn no tokens until something wakes you. You wake when a message
    arrives for you, when the timeout expires, or when the operator resumes
    you. Set a status so others know whether to disturb you.
    """,
    obj(
        {
            "seconds": {"type": "number", "description": "Optional timeout. Omit to sleep until a message arrives."},
            "status": {"type": "string", "description": "Short status shown on the board while you sleep."},
        }
    ),
    ends_turn=True,
)
def sleep(ctx: AgentContext, args: dict) -> ToolResult:
    seconds = args.get("seconds")
    raise Sleep(float(seconds) if seconds is not None else None, args.get("status", ""))


@tool(
    "finish",
    """
    Declare yourself done. You cannot be woken by messages afterwards -- only
    the operator can restart you. Use sleep() if other agents might still need
    you.
    """,
    obj({"summary": {"type": "string", "description": "What you accomplished."}}),
    ends_turn=True,
)
def finish(ctx: AgentContext, args: dict) -> ToolResult:
    raise Finish(args.get("summary", ""))
