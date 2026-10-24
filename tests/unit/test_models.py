import json

import pytest
from mercari_shopper.domain.models import (
    ItemStatus,
    Listing,
    Recommendation,
    SearchQuery,
    SortBy,
    SortOrder,
    condition_label,
    item_url,
)
from pydantic import ValidationError


def test_search_query_defaults() -> None:
    query = SearchQuery()
    assert query.keyword == ""
    assert query.status == [ItemStatus.ON_SALE]
    assert query.sort is SortBy.SCORE
    assert query.order is SortOrder.DESC
    assert query.page_size == 40


def test_search_query_rejects_bad_conditions_and_prices() -> None:
    with pytest.raises(ValidationError):
        SearchQuery(conditions=[0])
    with pytest.raises(ValidationError):
        SearchQuery(conditions=[7])
    with pytest.raises(ValidationError):
        SearchQuery(price_min=-1)
    with pytest.raises(ValidationError):
        SearchQuery(price_min=5000, price_max=1000)
    with pytest.raises(ValidationError):
        SearchQuery(page_size=0)
    with pytest.raises(ValidationError):
        SearchQuery(page_size=121)


def test_search_query_cache_key_is_stable_json() -> None:
    query = SearchQuery(keyword="switch", price_max=30000, conditions=[3, 4])
    key = query.cache_key()
    payload = json.loads(key)
    assert payload["keyword"] == "switch"
    assert payload["price_max"] == 30000
    assert query.cache_key() == key


def test_condition_label_and_item_url() -> None:
    assert condition_label(None) == "unknown"
    assert condition_label(1) == "new / unused"
    assert condition_label(99) == "unknown"
    assert item_url("m123", False) == "https://jp.mercari.com/item/m123"
    assert item_url("s456", True) == "https://jp.mercari.com/shops/product/s456"


def test_listing_fills_url_when_missing() -> None:
    listing = Listing(item_id="m123", name="Switch", price=20000)
    assert listing.url == "https://jp.mercari.com/item/m123"
    shop = Listing(item_id="s1", name="Shop", price=1, is_shop_item=True, url="")
    assert shop.url == "https://jp.mercari.com/shops/product/s1"
    custom = Listing(item_id="m9", name="x", price=1, url="https://custom.example/x")
    assert custom.url == "https://custom.example/x"


def test_recommendation_defaults() -> None:
    rec = Recommendation(item_id="m123", rank=1, reason="fits the budget")
    assert rec.pros == []
    assert rec.cons == []
