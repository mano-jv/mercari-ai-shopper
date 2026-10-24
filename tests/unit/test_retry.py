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


async def _noop_sleep(_delay: float) -> None:
    return None


class _FixedRng:
    def __init__(self, value: float) -> None:
        self.value = value

    def uniform(self, _a: float, _b: float) -> float:
        return self.value
