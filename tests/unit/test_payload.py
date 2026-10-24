from mercari_shopper.domain.models import ItemStatus, SearchQuery
from mercari_shopper.sources.payload import ITEM_URL, build_search_body, item_request_url


def test_build_search_body_maps_query_fields() -> None:
    query = SearchQuery(
        keyword="switch OLED",
        exclude_keyword="junk",
        price_min=1000,
        price_max=30000,
        conditions=[3, 4],
        page_size=5,
        page_token="next",
    )
    body = build_search_body(query, session_id="sess", device_uuid="dev")
    condition = body["searchCondition"]
    assert isinstance(condition, dict)
    assert body["pageSize"] == 5
    assert body["pageToken"] == "next"
    assert body["searchSessionId"] == "sess"
    assert body["laplaceDeviceUuid"] == "dev"
    assert condition["keyword"] == "switch OLED"
    assert condition["excludeKeyword"] == "junk"
    assert condition["priceMin"] == 1000
    assert condition["priceMax"] == 30000
    assert condition["itemConditionId"] == [3, 4]
    assert condition["status"] == [ItemStatus.ON_SALE.value]


def test_sold_search_adds_trading_once() -> None:
    sold = SearchQuery(keyword="switch", status=[ItemStatus.SOLD_OUT])
    statuses = build_search_body(sold, session_id="s", device_uuid="d")["searchCondition"]
    assert isinstance(statuses, dict)
    assert statuses["status"] == [ItemStatus.SOLD_OUT.value, ItemStatus.TRADING.value]

    both = SearchQuery(keyword="switch", status=[ItemStatus.SOLD_OUT, ItemStatus.TRADING])
    again = build_search_body(both, session_id="s", device_uuid="d")["searchCondition"]
    assert isinstance(again, dict)
    assert again["status"] == [ItemStatus.SOLD_OUT.value, ItemStatus.TRADING.value]


def test_missing_prices_become_zero() -> None:
    condition = build_search_body(SearchQuery(keyword="a"), session_id="s", device_uuid="d")[
        "searchCondition"
    ]
    assert isinstance(condition, dict)
    assert condition["priceMin"] == 0
    assert condition["priceMax"] == 0


def test_item_request_url() -> None:
    url = item_request_url("m123")
    assert url.startswith(ITEM_URL)
    assert "id=m123" in url
    assert "include_item_attributes=true" in url
    assert "include_auction=true" in url
