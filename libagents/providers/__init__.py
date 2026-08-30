from __future__ import annotations

from ..models import RunnerConfig
from .base import Provider, ToolCall, Turn  # noqa: F401
from .openai_provider import OpenAIProvider
from .openrouter_provider import OpenRouterProvider

PROVIDERS = {"openai": OpenAIProvider, "openrouter": OpenRouterProvider}


def make_provider(config: RunnerConfig) -> Provider:
    try:
        cls = PROVIDERS[config.provider]
    except KeyError as exc:
        raise KeyError(f"unknown provider: {config.provider}") from exc
    return cls(config)
