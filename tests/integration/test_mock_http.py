"""Integration: signed client + parser over a mocked Mercari HTTP API."""

import json
from importlib.resources import files

import httpx
import pytest
from mercari_shopper.domain.models import SearchQuery
from mercari_shopper.sources.api_source import DpopApiSource
from mercari_shopper.sources.retry import RateLimiter

pytestmark = pytest.mark.integration


def _json_fixture(name: str) -> dict[str, object]:
    raw = files("mercari_shopper.fixtures").joinpath(name).read_text(encoding="utf-8")
    payload = json.loads(raw)
    assert isinstance(payload, dict)
    return payload


async def test_search_then_item_detail_over_mock_http() -> None:
    search_payload = _json_fixture("search.json")
    item_payloads = _json_fixture("items.json")
    seen_methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_methods.append(f"{request.method} {request.url.path}")
        assert request.headers.get("x-platform") == "web"
        assert request.headers.get("dpop")
        if request.url.path.endswith("entities:search"):
            return httpx.Response(200, json=search_payload)
        item_id = request.url.params.get("id")
        payload = item_payloads.get(item_id) if item_id else None
        if not isinstance(payload, dict):
            return httpx.Response(404, text="gone")
        return httpx.Response(200, json=payload)

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client,
        DpopApiSource(
            client=client,
            limiter=RateLimiter(min_interval=0, jitter=0),
            retry_attempts=1,
        ) as source,
    ):
        result = await source.search(SearchQuery(keyword="switch", page_size=5))
        assert len(result.items) == 3
        details = await source.get_items(["m10000000001", "shopItem001abc"])
        assert len(details) == 1
        assert details[0].seller is not None
        assert details[0].description
        assert details[0].auction is None

    assert any(path.endswith("entities:search") for path in seen_methods)
    assert any("items/get" in path for path in seen_methods)
