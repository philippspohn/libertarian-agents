from __future__ import annotations

from .base import AgentContext, ToolError, ToolResult, obj, tool


@tool(
    "web_search",
    "Search the live web using the current model provider's hosted search tool. "
    "OpenAI uses Responses web_search; OpenRouter uses openrouter:web_search.",
    obj({"query": {"type": "string"}}, ["query"]),
)
def web_search(ctx: AgentContext, args: dict) -> ToolResult:
    # Provider adapters replace this logical tool with their hosted tool. This
    # function is a guard for a malformed provider response, not the normal
    # execution path.
    raise ToolError("web_search must be executed by the model provider")
