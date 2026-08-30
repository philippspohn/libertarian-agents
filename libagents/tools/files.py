from __future__ import annotations

import json
from pathlib import Path

from ..llm import quick_call
from .base import AgentContext, ToolError, ToolResult, obj, tool

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
    Read a bounded part of a file verbatim. Select a line range normally. If
    the runner's character cap cuts that range short, the result gives an
    exact start_char offset for the next call. start_char is also useful for
    continuing through a single very long line.
    """,
    obj(
        {
            "path": {"type": "string"},
            "start_line": {"type": "integer", "description": "1-indexed, default 1."},
            "num_lines": {"type": "integer", "description": "Default 400."},
            "start_char": {
                "type": "integer",
                "description": "0-indexed absolute character offset. When set, ignores the line range.",
            },
            "max_chars": {
                "type": "integer",
                "description": "Optional smaller content limit; cannot exceed the runner's configured cap.",
            },
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
        text = "\n".join(entries)
        source = f"{args['path']} (directory listing)"
    else:
        text = fp.read_text(encoding="utf-8", errors="replace")
        source = args["path"]

    line_parts = text.splitlines(keepends=True)
    line_offsets = [0]
    for part in line_parts:
        line_offsets.append(line_offsets[-1] + len(part))

    if args.get("start_char") is not None:
        start_offset = max(0, min(len(text), int(args["start_char"])))
        requested_end = len(text)
        line_mode = False
    else:
        start_line = max(1, int(args.get("start_line") or 1))
        count = max(1, int(args.get("num_lines") or 400))
        start_index = min(start_line - 1, len(line_parts))
        end_index = min(start_index + count, len(line_parts))
        start_offset = line_offsets[start_index]
        requested_end = line_offsets[end_index]
        line_mode = True

    configured_cap = max(1, int(ctx.config.read_file_char_limit))
    requested_max = args.get("max_chars")
    requested_cap = configured_cap if requested_max is None else max(1, int(requested_max))
    content_cap = min(configured_cap, requested_cap)
    end_offset = min(requested_end, start_offset + content_cap)
    chunk = text[start_offset:end_offset]

    line_number = text.count("\n", 0, start_offset) + 1
    previous_newline = text.rfind("\n", 0, start_offset)
    column = start_offset - previous_newline
    header = (
        f"{source} chars [{start_offset}:{end_offset}) of {len(text)}; "
        f"starts at line {line_number}, column {column}"
    )
    body = chunk if chunk else "(empty range)"

    if end_offset < requested_end:
        next_args = {"path": args["path"], "start_char": end_offset}
        continuation = (
            f"\n\n[read_file content cap {content_cap} chars reached; "
            f"continue with {json.dumps(next_args)}]"
        )
    elif line_mode and end_offset < len(text):
        next_args = {
            "path": args["path"],
            "start_line": text.count("\n", 0, end_offset) + 1,
        }
        continuation = f"\n\n[requested range complete; continue with {json.dumps(next_args)}]"
    else:
        continuation = ""

    return ToolResult(f"{header}\n{body}{continuation}")


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
    provider = ctx.config.summary_provider
    model = ctx.config.summary_model
    summary, usage = quick_call(
        provider,
        model,
        f"Summarize the following file in at most 15 short lines. Focus on {focus}.\n\n"
        f"--- {args['path']} ---\n{text}",
        max_output_tokens=ctx.output_allowance(1500),
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
