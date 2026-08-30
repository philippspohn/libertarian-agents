from __future__ import annotations

from .base import AgentContext, ToolError, ToolResult, obj, tool


@tool(
    "list_agents",
    "List everyone on the board with their state (active/waiting/inactive/finished) and self-set status.",
    obj({}),
)
def list_agents(ctx: AgentContext, args: dict) -> ToolResult:
    rows = ctx.board.list_agents()
    if not rows:
        return ToolResult("(nobody registered)")
    lines = [
        f"{r['name']:<16} {r['state']:<9} {r['status'] or '-'}"
        for r in rows
    ]
    return ToolResult("\n".join(lines))


@tool(
    "set_status",
    """
    Set your own status line, visible to everyone via list_agents. Use it to
    say what you are working on and whether you want to be messaged.
    """,
    obj({"status": {"type": "string"}}, ["status"]),
)
def set_status(ctx: AgentContext, args: dict) -> ToolResult:
    status = args["status"][:200]
    ctx.board.set_status(ctx.profile, status=status)
    return ToolResult(f"status set: {status}")


@tool(
    "send_message",
    """
    Send a message. Target is '#channel' for a channel (created on first use)
    or '@agent' for a direct message. Long messages are truncated on the board
    and the full text is written to a file the recipient can read. To prevent
    replies based on stale context, a send is normally blocked if that same
    channel or DM has unread inbound messages: the next page is returned and
    marked read so you can reconsider and retry. In a continuously busy scope,
    set send_anyway=true after reviewing the returned messages.
    """,
    obj(
        {
            "to": {"type": "string"},
            "body": {"type": "string"},
            "send_anyway": {
                "type": "boolean",
                "description": "Bypass the unread-message guard after reviewing the scope.",
            },
        },
        ["to", "body"],
    ),
)
def send_message(ctx: AgentContext, args: dict) -> ToolResult:
    try:
        if not args.get("send_anyway", False):
            unseen, more = ctx.board.fetch_unread_scope(ctx.profile, args["to"], limit=5)
            if unseen:
                suffix = (
                    "\nMore unread messages remain in this scope. Retry to review the next page, "
                    "or use send_anyway=true if your message is still valid."
                    if more
                    else
                    "\nReview these messages, then retry the send if your message is still valid."
                )
                return ToolResult(
                    f"MESSAGE NOT SENT: unread messages arrived in {args['to']}.\n"
                    + "\n".join(m.render() for m in unseen)
                    + suffix
                )
        mid, truncated = ctx.board.send(
            ctx.profile,
            args["to"],
            args["body"],
            max_chars=ctx.env_config.max_message_chars,
            spill_dir=ctx.profile_dir / "outputs",
        )
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    note = " (truncated; full text spilled to a file)" if truncated else ""
    return ToolResult(f"sent #{mid} to {args['to']}{note}")


@tool(
    "check_inbox",
    "Read unread messages addressed to you or to channels you joined. Advances only the cursors for scopes whose messages are returned.",
    obj({"limit": {"type": "integer", "description": "Default 30."}}),
)
def check_inbox(ctx: AgentContext, args: dict) -> ToolResult:
    msgs = ctx.board.fetch_unread(ctx.profile, limit=int(args.get("limit") or 30))
    if not msgs:
        return ToolResult("(no unread messages)")
    return ToolResult("\n".join(m.render() for m in msgs))


@tool(
    "read_history",
    "Read recent messages in a channel ('#name') or a DM thread ('@agent'). Does not change your unread count.",
    obj({"scope": {"type": "string"}, "limit": {"type": "integer"}}),
)
def read_history(ctx: AgentContext, args: dict) -> ToolResult:
    msgs = ctx.board.history(ctx.profile, args.get("scope"), limit=int(args.get("limit") or 20))
    if not msgs:
        return ToolResult("(no messages)")
    return ToolResult("\n".join(m.render() for m in msgs))


@tool(
    "join_channel",
    "Join a channel, creating it if it does not exist. You then receive its messages in your inbox.",
    obj({"channel": {"type": "string"}, "topic": {"type": "string"}}, ["channel"]),
)
def join_channel(ctx: AgentContext, args: dict) -> ToolResult:
    name = args["channel"].lstrip("#")
    if not name:
        raise ToolError("channel name cannot be empty")
    ctx.board.ensure_channel(name, args.get("topic", ""), by=ctx.profile)
    ctx.board.subscribe(ctx.profile, name)
    return ToolResult(f"joined #{name}")


@tool(
    "leave_channel",
    "Leave a channel. You stop receiving its messages in your inbox; the channel and its history remain.",
    obj({"channel": {"type": "string"}}, ["channel"]),
)
def leave_channel(ctx: AgentContext, args: dict) -> ToolResult:
    name = args["channel"].lstrip("#")
    if not name:
        raise ToolError("channel name cannot be empty")
    if not ctx.board.unsubscribe(ctx.profile, name):
        return ToolResult(f"not subscribed to #{name}")
    return ToolResult(f"left #{name}")
