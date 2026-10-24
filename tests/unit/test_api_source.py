import asyncio
import json
from importlib.resources import files

import httpx
import pytest
from mercari_shopper.domain.models import SearchQuery
from mercari_shopper.sources.api_source import DpopApiSource
from mercari_shopper.sources.errors import (
    MercariHttpError,
    MercariParseError,
    MercariTransportError,
)
from mercari_shopper.sources.payload import SEARCH_URL
from mercari_shopper.sources.retry import RateLimiter


def _search_payload() -> dict[str, object]:
    raw = files("mercari_shopper.fixtures").joinpath("search.json").read_text(encoding="utf-8")
    payload = json.loads(raw)
    assert isinstance(payload, dict)
    return payload


def _client(handler: object) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _source(handler: object, *, attempts: int = 3) -> DpopApiSource:
    sleeps: list[float] = []

    async def sleep(_delay: float) -> None:
        sleeps.append(_delay)

    source = DpopApiSource(
        client=_client(handler),
        limiter=RateLimiter(min_interval=0, jitter=0),
        retry_attempts=attempts,
        retry_sleep=sleep,
    )
    source.sleeps = sleeps  # type: ignore[attr-defined]
    return source


async def test_search_signs_request_and_parses_fixture() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert str(request.url) == SEARCH_URL
        assert request.method == "POST"
        assert request.headers["x-platform"] == "web"
        assert request.headers["dpop"]
        body = json.loads(request.content)
        assert body["searchCondition"]["keyword"] == "switch 有機EL"
        return httpx.Response(200, json=_search_payload())

    source = _source(handler)
    result = await source.search(SearchQuery(keyword="switch 有機EL", page_size=5))
    await source.aclose()
    assert len(result.items) == 3
    assert result.items[0].item_id == "m10000000001"
    assert seen[0].headers["dpop"]


async def test_empty_results() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"meta": {"numFound": "0"}, "items": []})

    source = _source(handler)
    result = await source.search(SearchQuery(keyword="zzzz"))
    await source.aclose()
    assert result.items == []
    assert result.num_found == 0


@pytest.mark.parametrize("status", [400, 401, 403, 429])
async def test_client_errors_are_not_retried(status: int) -> None:
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        headers = {"Retry-After": "2.5"} if status == 429 else None
        return httpx.Response(status, text="nope", headers=headers)

    source = _source(handler, attempts=3)
    with pytest.raises(MercariHttpError) as raised:
        await source.search(SearchQuery(keyword="switch"))
    await source.aclose()
    assert raised.value.status == status
    assert calls["n"] == 1
    if status == 429:
        assert raised.value.retry_after == 2.5


async def test_5xx_is_retried_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="down")
        return httpx.Response(200, json=_search_payload())

    source = _source(handler, attempts=3)
    result = await source.search(SearchQuery(keyword="switch"))
    await source.aclose()
    assert calls["n"] == 3
    assert len(result.items) == 3
    assert source.sleeps == [0.4, 0.8]  # type: ignore[attr-defined]


async def test_malformed_json() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not-json")

    source = _source(handler)
    with pytest.raises(MercariParseError):
        await source.search(SearchQuery(keyword="switch"))
    await source.aclose()


async def test_get_items_skips_404() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        item_id = request.url.params.get("id")
        if item_id == "missing":
            return httpx.Response(404, text="gone")
        return httpx.Response(
            200,
            json={
                "data": {
                    "id": item_id,
                    "name": "Switch",
                    "price": 1000,
                    "status": "on_sale",
                    "item_condition": {"id": 3, "name": "ok"},
                }
            },
        )

    source = _source(handler)
    listings = await source.get_items(["m1", "missing"])
    await source.aclose()
    assert [item.item_id for item in listings] == ["m1"]


async def test_get_items_dedupes_ids() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        item_id = request.url.params.get("id")
        return httpx.Response(
            200,
            json={"data": {"id": item_id, "name": "Switch", "price": 1000, "status": "on_sale"}},
        )

    source = _source(handler)
    listings = await source.get_items(["m1", "m1", "", "m1"])
    await source.aclose()
    assert calls["n"] == 1
    assert [item.item_id for item in listings] == ["m1"]


async def test_get_items_concurrency_preserves_requested_order() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        item_id = request.url.params.get("id") or ""
        return httpx.Response(
            200,
            json={"data": {"id": item_id, "name": item_id, "price": 1000, "status": "on_sale"}},
        )

    source = DpopApiSource(
        client=_client(handler),
        limiter=RateLimiter(min_interval=0, jitter=0, max_concurrent=3),
    )
    listings = await source.get_items(["c", "a", "b"])
    await source.aclose()
    # Requested order is preserved regardless of handler completion order.
    assert [item.item_id for item in listings] == ["c", "a", "b"]


async def test_get_items_bounds_concurrency_to_limiter_max_concurrent() -> None:
    in_flight = 0
    peak_in_flight = 0

    async def fake_get_item(item_id: str) -> None:
        nonlocal in_flight, peak_in_flight
        in_flight += 1
        peak_in_flight = max(peak_in_flight, in_flight)
        await asyncio.sleep(0)  # yield so other pending fetches can start
        in_flight -= 1
        return None

    source = DpopApiSource(limiter=RateLimiter(min_interval=0, jitter=0, max_concurrent=2))
    source._get_item = fake_get_item  # type: ignore[method-assign]
    await source.get_items(["a", "b", "c", "d", "e"])
    assert peak_in_flight <= 2


async def test_transport_error_becomes_mercari_transport_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    source = _source(handler, attempts=1)
    with pytest.raises(MercariTransportError):
        await source.search(SearchQuery(keyword="switch"))
    await source.aclose()
