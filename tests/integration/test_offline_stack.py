"""Integration: factory + cache + fixtures, no network."""

import pytest
from mercari_shopper.domain.models import SearchQuery
from mercari_shopper.sources.factory import build_source

pytestmark = pytest.mark.integration


async def test_fixture_stack_search_then_details_and_cache() -> None:
    async with build_source("fixture") as source:
        result = await source.search(SearchQuery(keyword="switch", page_size=5))
        assert len(result.items) == 3
        assert {item.item_id for item in result.items} >= {"m10000000001", "m10000000002"}

        again = await source.search(SearchQuery(keyword="switch", page_size=5))
        assert [item.item_id for item in again.items] == [item.item_id for item in result.items]

        details = await source.get_items(["m10000000001", "m10000000001"])
        assert len(details) == 1
        assert details[0].item_id == "m10000000001"
        assert details[0].seller is not None
        assert details[0].seller.good >= 0


async def test_fixture_stack_respects_price_filter() -> None:
    async with build_source("fixture") as source:
        cheap = await source.search(SearchQuery(keyword="switch", price_max=25000))
        assert all(item.price <= 25000 for item in cheap.items)
        assert cheap.items
