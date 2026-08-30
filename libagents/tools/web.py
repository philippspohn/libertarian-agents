from __future__ import annotations

from ..llm import web_search_call
from .base import AgentContext, ToolError, ToolResult, obj, tool


@tool(
    "web_search",
    "Search the web. Returns a short digest with source URLs, not raw pages.",
    obj({"query": {"type": "string"}}, ["query"]),
)
def web_search(ctx: AgentContext, args: dict) -> ToolResult:
    model = ctx.env_config.summary_model
    try:
        digest, usage = web_search_call(model, args["query"])
    except Exception as exc:  # network/provider failures are recoverable
        raise ToolError(f"web search failed: {exc}") from exc
    ctx.add_usage(model, usage)
    if len(digest) > 4000:
        return ToolResult(digest[:4000] + f"\n... full: {ctx.spill(digest, 'search')}", full=digest)
    return ToolResult(digest or "(no results)")
