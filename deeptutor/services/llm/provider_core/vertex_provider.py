"""Vertex AI provider — direct SDK integration for Google Gemini models.

Uses the google-genai SDK for native Vertex AI support.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, AsyncGenerator

from deeptutor.services.llm.provider_core.base import LLMProvider, LLMResponse, ToolCallRequest

class VertexAIProvider(LLMProvider):
    """LLM provider using the google-genai SDK for Vertex AI."""

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str = "gemini-3.1-pro-preview",
        extra_headers: dict[str, str] | None = None,
        spec: Any = None,
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self.extra_headers = extra_headers or {}
        self._spec = spec

    @classmethod
    def _handle_error(cls, e: Exception) -> LLMResponse:
        return LLMResponse(content=f"Error calling LLM: {e}", finish_reason="error")

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **extra_kwargs: Any,
    ) -> LLMResponse:
        from deeptutor.services.llm.vertex_auth import vertex_complete
        
        # Note: tools and tool_choice are not yet fully integrated in vertex_auth.py
        # but we can pass them in extra_kwargs if vertex_complete supports them.
        
        model_name = model or self.default_model
        
        # BaseAgent usually sends system_prompt and user_prompt or messages.
        # factory.py already built messages list.
        
        # vertex_complete expects prompt and system_prompt.
        # We can extract them from messages if available.
        system_prompt = "You are a helpful assistant."
        user_prompt = ""
        
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "user":
                user_prompt = msg["content"]
        
        try:
            content = await vertex_complete(
                prompt=user_prompt,
                system_prompt=system_prompt,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                **extra_kwargs,
            )
            return LLMResponse(content=content, finish_reason="stop")
        except Exception as e:
            return self._handle_error(e)

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
        on_reasoning_delta: Callable[[str], Awaitable[None]] | None = None,
        **extra_kwargs: Any,
    ) -> LLMResponse:
        from deeptutor.services.llm.vertex_auth import vertex_stream
        
        model_name = model or self.default_model
        
        full_content = ""
        try:
            async for chunk in vertex_stream(
                prompt="", # Not used if messages provided
                system_prompt="", # Not used if messages provided
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **extra_kwargs,
            ):
                full_content += chunk
                if on_content_delta:
                    await on_content_delta(chunk)
            
            return LLMResponse(content=full_content, finish_reason="stop")
        except Exception as e:
            return self._handle_error(e)

    def get_default_model(self) -> str:
        return self.default_model
