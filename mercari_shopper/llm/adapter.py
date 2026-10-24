"""Template Method shared by provider adapters.

Subclasses only map the request and parse the response. Retries stay in the
agent loop, which branches on LLMError.retryable.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Sequence
from time import monotonic
from typing import Any

from mercari_shopper.llm.base import (
    LLMResponse,
    Message,
    ToolChoice,
    ToolChoiceMode,
    ToolSpec,
    check_transcript,
    guard_tool_choice,
)
from mercari_shopper.llm.errors import LLMError, translate_provider_error
from mercari_shopper.logging import log_event


class LLMAdapter(ABC):
    """Adapter base. `complete` is the template; providers fill in three steps."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        max_tokens: int,
        timeout: float,
        client: Any,
        owns_client: bool,
        effort: str = "",
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        stripped = model.strip()
        if not stripped:
            raise ValueError("model is empty")
        self.provider = provider
        self._model = stripped
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._client = client
        self._owns_client = owns_client
        # "" means don't send a reasoning-effort param; let the model/provider
        # default apply. Lower effort trims time spent reasoning before the
        # first visible token, which matters most on turn 1 (no cache yet).
        self._effort = effort.strip().lower()

    @property
    def model(self) -> str:
        return self._model

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = (),
        tool_choice: ToolChoice = ToolChoiceMode.AUTO,
    ) -> LLMResponse:
        check_transcript(messages)
        tool_list = tuple(tools)
        guard_tool_choice(tool_choice, tool_list)
        payload = self.build_payload(messages, tools=tool_list, tool_choice=tool_choice)
        started = monotonic()
        try:
            raw = await self.send(payload)
        except LLMError:
            raise
        except Exception as exc:
            raise translate_provider_error(exc, provider=self.provider) from exc
        response = self.parse(raw)
        log_event(
            "llm_complete",
            provider=self.provider,
            model=self._model,
            stop_reason=response.stop_reason,
            tool_calls=len(response.tool_calls),
            elapsed_s=round(monotonic() - started, 3),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cache_read_tokens=response.cache_read_tokens,
            cache_write_tokens=response.cache_write_tokens,
        )
        return response

    @abstractmethod
    def build_payload(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec],
        tool_choice: ToolChoice,
    ) -> dict[str, Any]:
        """Return the kwargs for this provider's create call."""

    @abstractmethod
    async def send(self, payload: dict[str, Any]) -> Any:
        """Perform the SDK call."""

    @abstractmethod
    def parse(self, raw: Any) -> LLMResponse:
        """Turn the SDK response into a normalized assistant message."""

    async def aclose(self) -> None:
        if not self._owns_client:
            return
        close = getattr(self._client, "close", None)
        if close is None:
            return
        result = close()
        if inspect.isawaitable(result):
            await result
