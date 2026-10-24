import asyncio

import pytest
from mercari_shopper.sources.errors import MercariHttpError, MercariTransportError
from mercari_shopper.sources.retry import RateLimiter, retry_async


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


async def test_retry_retries_5xx_then_succeeds() -> None:
    sleeps: list[float] = []
    calls = {"n": 0}

    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise MercariHttpError(503, "down")
        return "ok"

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    assert await retry_async(flaky, attempts=3, base_delay=0.5, sleep=sleep) == "ok"
    assert calls["n"] == 3
    assert sleeps == [0.5, 1.0]


@pytest.mark.parametrize("status", [401, 403, 429])
async def test_retry_does_not_retry_auth_or_rate_limit(status: int) -> None:
    calls = {"n": 0}

    async def boom() -> str:
        calls["n"] += 1
        raise MercariHttpError(status, "nope")

    with pytest.raises(MercariHttpError) as raised:
        await retry_async(boom, attempts=3, sleep=_noop_sleep)
    assert raised.value.status == status
    assert calls["n"] == 1


async def test_retry_transport_then_raise() -> None:
    async def boom() -> str:
        raise MercariTransportError("offline")

    with pytest.raises(MercariTransportError):
        await retry_async(boom, attempts=2, sleep=_noop_sleep)


async def test_rate_limiter_waits_for_interval_plus_jitter() -> None:
    clock = Clock()
    sleeps: list[float] = []

    async def sleep(delay: float) -> None:
        sleeps.append(delay)
        clock.now += delay

    limiter = RateLimiter(
        min_interval=1.0,
        jitter=0.5,
        sleep=sleep,
        clock=clock,
        rng=_FixedRng(0.2),
    )
    await limiter.wait()
    assert sleeps == [1.2]
    await limiter.wait()
    assert sleeps == [1.2, 1.2]


async def test_rate_limiter_defaults_to_max_concurrent_one() -> None:
    assert RateLimiter().max_concurrent == 1


async def test_rate_limiter_rejects_non_positive_max_concurrent() -> None:
    with pytest.raises(ValueError, match="max_concurrent"):
        RateLimiter(max_concurrent=0)


async def test_rate_limiter_allows_a_burst_up_to_max_concurrent() -> None:
    """With max_concurrent=2, two concurrent waiters return instantly (a real
    monotonic clock is always far past the 0.0 sentinel); only the third
    waits for the gap measured from the first of the two."""
    clock = Clock()
    clock.now = 1_000.0  # simulate a real monotonic clock, not the 0.0 sentinel
    sleeps: list[float] = []

    async def sleep(delay: float) -> None:
        sleeps.append(delay)
        clock.now += delay

    limiter = RateLimiter(
        min_interval=1.0,
        jitter=0.0,
        max_concurrent=2,
        sleep=sleep,
        clock=clock,
        rng=_FixedRng(0.0),
    )
    await asyncio.gather(limiter.wait(), limiter.wait())
    assert sleeps == []  # both started immediately: within the burst allowance

    await limiter.wait()
    assert sleeps == [1.0]  # third call waits for the gap from the first of the two


async def test_rate_limiter_concurrent_waiters_do_not_block_each_others_sleep() -> None:
    """A waiter forced to sleep must release the lock first. Otherwise a
    second waiter would stay stuck at the lock instead of reaching its own
    sleep call, and `sleep_calls` would stop at one entry below."""
    clock = Clock()
    clock.now = 1_000.0
    gate = asyncio.Event()
    sleep_calls: list[float] = []

    async def sleep(delay: float) -> None:
        sleep_calls.append(delay)
        await gate.wait()
        clock.now += delay

    limiter = RateLimiter(
        min_interval=1.0,
        jitter=0.0,
        max_concurrent=1,
        sleep=sleep,
        clock=clock,
        rng=_FixedRng(0.0),
    )
    await limiter.wait()  # 0.0 sentinel is far in the past: no sleep yet
    assert sleep_calls == []

    first = asyncio.create_task(limiter.wait())
    await asyncio.sleep(0)
    assert sleep_calls == [1.0]  # first reached its sleep call and is now blocked on the gate

    second = asyncio.create_task(limiter.wait())
    await asyncio.sleep(0)
    # Proves the lock was released before the sleep: the second waiter
    # reached its own sleep call instead of being stuck at the lock.
    assert sleep_calls == [1.0, 1.0]

    gate.set()
    await asyncio.wait_for(asyncio.gather(first, second), timeout=1.0)


async def _noop_sleep(_delay: float) -> None:
    return None


class _FixedRng:
    def __init__(self, value: float) -> None:
        self.value = value

    def uniform(self, _a: float, _b: float) -> float:
        return self.value
