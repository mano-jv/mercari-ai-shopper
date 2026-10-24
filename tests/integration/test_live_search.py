"""Optional live smoke tests. Skipped unless pytest is run with -m live."""

import pytest
from mercari_shopper.domain.models import SearchQuery
from mercari_shopper.sources.api_source import DpopApiSource
from mercari_shopper.sources.retry import RateLimiter

pytestmark = [pytest.mark.live, pytest.mark.integration]


async def test_dpop_live_search_and_item_detail() -> None:
    source = DpopApiSource(limiter=RateLimiter(min_interval=1.0, jitter=0.2), retry_attempts=2)
    async with source:
        result = await source.search(SearchQuery(keyword="switch", page_size=3))
        assert result.num_found >= 0
        assert len(result.items) <= 3
        for listing in result.items:
            assert listing.item_id
            assert listing.name
            assert listing.price >= 0
            assert listing.url.startswith("https://jp.mercari.com/")

        mercari_ids = [item.item_id for item in result.items if item.item_id.startswith("m")]
        if not mercari_ids:
            return
        details = await source.get_items(mercari_ids[:1])
        if not details:
            return
        detail = details[0]
        assert detail.item_id == mercari_ids[0]
        assert detail.name
        assert detail.price >= 0
