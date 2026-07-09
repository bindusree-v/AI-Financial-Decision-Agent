"""
LLM factory — returns a configured ChatOpenAI instance.
Supports OpenRouter (recommended) or direct OpenAI.
"""
from __future__ import annotations

from langchain_openai import ChatOpenAI

from config import config


def get_llm(temperature: float | None = None) -> ChatOpenAI:
    """
    Build and return a ChatOpenAI instance.
    Uses OpenRouter if OPENROUTER_API_KEY is set, otherwise falls back to OpenAI.
    """
    base_url = config.get_llm_base_url()
    api_key = config.get_llm_api_key()
    temp = temperature if temperature is not None else config.LLM_TEMPERATURE

    kwargs: dict = {
        "model": config.LLM_MODEL,
        "temperature": temp,
        "api_key": api_key,
        "max_tokens": config.LLM_MAX_TOKENS,
    }
    if base_url:
        kwargs["base_url"] = base_url
        # OpenRouter requires this header
        kwargs["default_headers"] = {
            "HTTP-Referer": "https://financial-research-agent",
            "X-Title": "Financial Deep Research Agent",
        }

    return ChatOpenAI(**kwargs)
