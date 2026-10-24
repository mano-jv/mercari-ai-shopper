import json
from importlib.resources import files

import pytest
from mercari_shopper.domain.models import SearchQuery
from mercari_shopper.sources.browser_source import BrowserSource
from mercari_shopper.sources.errors import MercariHttpError
from mercari_shopper.sources.payload import build_search_page_url
from mercari_shopper.sources.retry import RateLimiter


def _search_payload() -> dict[str, object]:
    raw = files("mercari_shopper.fixtures").joinpath("search.json").read_text(encoding="utf-8")
    payload = json.loads(raw)
    assert isinstance(payload, dict)
    return payload


async def test_browser_source_uses_intercepted_json() -> None:
    seen: list[tuple[str, str]] = []

    async def intercept(page_url: str, fragment: str) -> tuple[int, object]:
        seen.append((page_url, fragment))
        return 200, _search_payload()

    source = BrowserSource(limiter=RateLimiter(min_interval=0, jitter=0), intercept=intercept)
    query = SearchQuery(keyword="switch 有機EL", price_max=30000, conditions=[3])
    result = await source.search(query)
    assert len(result.items) == 3
    assert seen[0][1] == "entities:search"
    assert seen[0][0] == build_search_page_url(query)
    assert "keyword=switch" in seen[0][0]


async def test_browser_source_honors_page_size() -> None:
    async def intercept(_page_url: str, _fragment: str) -> tuple[int, object]:
        return 200, _search_payload()

    source = BrowserSource(limiter=RateLimiter(min_interval=0, jitter=0), intercept=intercept)
    result = await source.search(SearchQuery(keyword="switch", page_size=1))
    assert len(result.items) == 1
    assert result.items[0].item_id == "m10000000001"


async def test_browser_source_raises_http_error() -> None:
    async def intercept(_page_url: str, _fragment: str) -> tuple[int, object]:
        return 403, {}

    source = BrowserSource(limiter=RateLimiter(min_interval=0, jitter=0), intercept=intercept)
    with pytest.raises(MercariHttpError) as raised:
        await source.search(SearchQuery(keyword="switch"))
    assert raised.value.status == 403
