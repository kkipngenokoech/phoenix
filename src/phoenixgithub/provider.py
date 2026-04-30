"""LLM provider factory — creates the right LangChain chat model."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from phoenixgithub.config import LLMConfig, LLMProvider


def create_llm(config: LLMConfig) -> BaseChatModel:
    kwargs: dict = {
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }

    if config.provider == LLMProvider.ANTHROPIC:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config.model,
            api_key=config.api_key,
            **({"base_url": config.base_url} if config.base_url else {}),
            **kwargs,
        )

    if config.provider == LLMProvider.OPENAI:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.model,
            api_key=config.api_key,
            **({"base_url": config.base_url} if config.base_url else {}),
            **kwargs,
        )

    raise ValueError(f"Unsupported LLM provider: {config.provider}")
