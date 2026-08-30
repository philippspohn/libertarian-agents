from __future__ import annotations

import shlex

from .base import AgentContext, ToolResult, compress, obj, tool


@tool(
    "shell",
    """
    Run a shell command in your persistent per-agent shell session. The
    working directory, exported variables, shell functions, and background
    jobs survive into later shell calls until the environment/server stops.
    The initial directory is your own agent folder; `cwd` changes it before
    this command. Output is heavily compressed: you get
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
    command = args["command"]
    if args.get("cwd"):
        try:
            target = ctx.mapper.to_env(ctx.mapper.to_host(args["cwd"]))
        except ValueError as exc:
            return ToolResult(f"ERROR: {exc}")
        command = f"cd {shlex.quote(target)} && {command}"
    timeout = int(args.get("timeout") or cfg.shell_timeout)
    result = ctx.sandbox.exec(
        command, cwd=ctx.cwd(), timeout=timeout, session=ctx.profile
    )

    body, elided = compress(result.output, cfg.shell_head_lines, cfg.shell_tail_lines, cfg.shell_line_chars)
    parts = [f"exit={result.exit_code}" + (" (timed out)" if result.timed_out else "")]
    if body:
        parts.append(body)
    if elided:
        parts.append(f"full output: {ctx.spill(result.output, 'shell')}")
    elif not body:
        parts.append("(no output)")
    return ToolResult("\n".join(parts), full=result.output)
