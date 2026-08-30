from __future__ import annotations

from pathlib import Path

from ..llm import quick_call
from .base import AgentContext, ToolError, ToolResult, obj, tool

MAX_READ_CHARS = 60_000
"""read_file is deliberately the one tool allowed to return a lot of text."""


def _resolve(ctx: AgentContext, path: str) -> Path:
    try:
        return ctx.mapper.to_host(path)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc


def _guard_memory(ctx: AgentContext, target: Path, new_text: str) -> None:
    """memory.md is the state snapshot re-injected after every compaction, so
    it has a hard size limit. Fail the edit rather than silently truncating --
    the agent should decide what to drop."""
    if target.resolve() != (ctx.profile_dir / "memory.md").resolve():
        return
    limit = ctx.config.memory_char_limit
    if len(new_text) > limit:
        raise ToolError(
            f"memory.md would be {len(new_text)} chars, limit is {limit}. "
            f"Reduce it by ~{len(new_text) - limit} chars and retry. Nothing was written."
        )


@tool(
    "read_file",
    """
    Read a file, optionally a line range. This is the one tool that returns
    long output verbatim, so use it when you actually need the detail.
    """,
    obj(
        {
            "path": {"type": "string"},
            "start_line": {"type": "integer", "description": "1-indexed, default 1."},
            "num_lines": {"type": "integer", "description": "Default 400."},
        },
        ["path"],
    ),
)
def read_file(ctx: AgentContext, args: dict) -> ToolResult:
    fp = _resolve(ctx, args["path"])
    if not fp.exists():
        raise ToolError(f"no such file: {args['path']}")
    if fp.is_dir():
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in fp.iterdir())
        return ToolResult(f"{args['path']} is a directory:\n" + "\n".join(entries[:200]))

    text = fp.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    start = max(1, int(args.get("start_line") or 1))
    count = max(1, int(args.get("num_lines") or 400))
    chunk = lines[start - 1: start - 1 + count]
    body = "\n".join(f"{start + i}\t{ln}" for i, ln in enumerate(chunk))
    if len(body) > MAX_READ_CHARS:
        body = body[:MAX_READ_CHARS] + "\n... [clipped, narrow the line range]"
    header = f"{args['path']} lines {start}-{start + len(chunk) - 1} of {len(lines)}"
    return ToolResult(f"{header}\n{body}")


@tool(
    "read_summary",
    "Summarize a file with a cheap model. Use for long files you only need the gist of.",
    obj({"path": {"type": "string"}, "focus": {"type": "string", "description": "What to pay attention to."}}, ["path"]),
)
def read_summary(ctx: AgentContext, args: dict) -> ToolResult:
    fp = _resolve(ctx, args["path"])
    if not fp.exists():
        raise ToolError(f"no such file: {args['path']}")
    text = fp.read_text(encoding="utf-8", errors="replace")
    focus = args.get("focus") or "the overall content and anything actionable"
    model = ctx.env_config.summary_model
    summary, usage = quick_call(
        model,
        f"Summarize the following file in at most 15 short lines. Focus on {focus}.\n\n"
        f"--- {args['path']} ---\n{text}",
    )
    ctx.add_usage(model, usage)
    return ToolResult(summary or "(empty summary)")


@tool(
    "write_file",
    "Create or overwrite a file with the given content.",
    obj({"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
)
def write_file(ctx: AgentContext, args: dict) -> ToolResult:
    fp = _resolve(ctx, args["path"])
    _guard_memory(ctx, fp, args["content"])
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(args["content"], encoding="utf-8")
    return ToolResult(f"wrote {len(args['content'])} chars to {args['path']}")


@tool(
    "edit_file",
    """
    Replace an exact string in a file. old_string must appear exactly once
    unless replace_all is set.
    """,
    obj(
        {
            "path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
            "replace_all": {"type": "boolean"},
        },
        ["path", "old_string", "new_string"],
    ),
)
def edit_file(ctx: AgentContext, args: dict) -> ToolResult:
    fp = _resolve(ctx, args["path"])
    if not fp.exists():
        raise ToolError(f"no such file: {args['path']}")
    text = fp.read_text(encoding="utf-8", errors="replace")
    old, new = args["old_string"], args["new_string"]
    hits = text.count(old)
    if hits == 0:
        raise ToolError("old_string not found; read the file and match it exactly")
    if hits > 1 and not args.get("replace_all"):
        raise ToolError(f"old_string appears {hits} times; add more context or set replace_all")
    updated = text.replace(old, new) if args.get("replace_all") else text.replace(old, new, 1)
    _guard_memory(ctx, fp, updated)
    fp.write_text(updated, encoding="utf-8")
    return ToolResult(f"edited {args['path']} ({hits if args.get('replace_all') else 1} replacement(s))")


@tool("delete_file", "Delete a file.", obj({"path": {"type": "string"}}, ["path"]))
def delete_file(ctx: AgentContext, args: dict) -> ToolResult:
    fp = _resolve(ctx, args["path"])
    if not fp.exists():
        raise ToolError(f"no such file: {args['path']}")
    if fp.is_dir():
        raise ToolError("refusing to delete a directory; use shell if you mean it")
    fp.unlink()
    return ToolResult(f"deleted {args['path']}")
