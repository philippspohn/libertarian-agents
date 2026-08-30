from __future__ import annotations

from .base import AgentContext, ToolResult, compress, obj, tool


@tool(
    "shell",
    """
    Run a shell command inside the environment sandbox. Working directory
    defaults to your own agent folder. Output is heavily compressed: you get
    the first and last few lines only, and the full output is written to a
    file whose path is returned -- use read_file or read_summary on it when
    you need more. Pipe through head/tail/grep to keep things small.
    """,
    obj(
        {
            "command": {"type": "string", "description": "Shell command to run."},
            "cwd": {"type": "string", "description": "Optional working directory."},
            "timeout": {"type": "integer", "description": "Seconds before the command is killed."},
        },
        ["command"],
    ),
)
def shell(ctx: AgentContext, args: dict) -> ToolResult:
    cfg = ctx.config
    cwd = args.get("cwd") or ctx.cwd()
    timeout = int(args.get("timeout") or cfg.shell_timeout)
    result = ctx.sandbox.exec(args["command"], cwd=cwd, timeout=timeout)

    body, elided = compress(result.output, cfg.shell_head_lines, cfg.shell_tail_lines, cfg.shell_line_chars)
    parts = [f"exit={result.exit_code}" + (" (timed out)" if result.timed_out else "")]
    if body:
        parts.append(body)
    if elided:
        parts.append(f"full output: {ctx.spill(result.output, 'shell')}")
    elif not body:
        parts.append("(no output)")
    return ToolResult("\n".join(parts), full=result.output)
