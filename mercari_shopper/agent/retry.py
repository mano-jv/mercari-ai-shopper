"""Retry retryable LLM errors. Non-retryable failures fail the loop immediately."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from mercari_shopper.agent.errors import AgentLLMError
from mercari_shopper.llm.errors import LLMError
from mercari_shopper.logging import log_event


async def retry_llm[T](
    operation: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 0.4,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Retry when `LLMError.retryable` is set. 4xx other than 429 is not retried."""
    last_error: LLMError | None = None
    total = max(1, attempts)
    for attempt in range(total):
        try:
            return await operation()
        except LLMError as error:
            last_error = error
            log_event(
                "llm_retry",
                retryable=error.retryable,
                status=error.status,
                attempt=attempt + 1,
                attempts=total,
            )
            if not error.retryable or attempt == total - 1:
                raise AgentLLMError(str(error)) from error
            await sleep(base_delay * (2**attempt))
    assert last_error is not None
    raise AgentLLMError(str(last_error)) from last_error
