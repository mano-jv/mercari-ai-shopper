"""Retry with exponential backoff, and a minimum gap between live requests."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable, Collection
from time import monotonic

from mercari_shopper.sources.errors import MercariHttpError, MercariTransportError

_RETRY_STATUSES = frozenset({500, 502, 503, 504})


async def retry_async[T](
    operation: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 0.4,
    retry_statuses: Collection[int] = _RETRY_STATUSES,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Retry 5xx and transport failures. 401/403/429 are not retried."""
    last_error: BaseException | None = None
    total = max(1, attempts)
    for attempt in range(total):
        try:
            return await operation()
        except MercariHttpError as error:
            last_error = error
            if error.status not in retry_statuses or attempt == total - 1:
                raise
            delay = (
                error.retry_after if error.retry_after is not None else base_delay * (2**attempt)
            )
            await sleep(delay)
        except MercariTransportError as error:
            last_error = error
            if attempt == total - 1:
                raise
            await sleep(base_delay * (2**attempt))
    assert last_error is not None
    raise last_error


class RateLimiter:
    """One shared gap, plus jitter, between Mercari calls on this process."""

    def __init__(
        self,
        min_interval: float = 1.0,
        jitter: float = 0.3,
        *,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        clock: Callable[[], float] = monotonic,
        rng: random.Random | None = None,
    ) -> None:
        self.min_interval = min_interval
        self.jitter = jitter
        self._sleep = sleep or asyncio.sleep
        self._clock = clock
        self._rng = rng or random.Random()
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def wait(self) -> None:
        async with self._lock:
            delay = self.min_interval
            if self.jitter > 0:
                delay += self._rng.uniform(0.0, self.jitter)
            if delay <= 0:
                self._last = self._clock()
                return
            now = self._clock()
            wait_for = self._last + delay - now
            if wait_for > 0:
                await self._sleep(wait_for)
            self._last = self._clock()
