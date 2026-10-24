import json
from datetime import UTC, datetime
from importlib.resources import files

import pytest
from mercari_shopper.sources.errors import MercariParseError
from mercari_shopper.sources.parser import parse_item_response, parse_search_response


def _package_json(name: str) -> dict[str, object]:
    raw = files("mercari_shopper.fixtures").joinpath(name).read_text(encoding="utf-8")
    payload = json.loads(raw)
    assert isinstance(payload, dict)
    return payload


def test_parse_search_regular_auction_and_shop_items() -> None:
    result = parse_search_response(_package_json("search.json"))
    assert result.num_found == 3
    assert result.next_page_token == "v1:1"
    by_id = {item.item_id: item for item in result.items}

    regular = by_id["m10000000001"]
    assert regular.price == 24800
    assert regular.status == "on_sale"
    assert regular.condition_id == 3
    assert regular.seller_id == "1001"
    assert regular.is_shop_item is False
    assert regular.url == "https://jp.mercari.com/item/m10000000001"
    assert regular.thumbnail_url and regular.thumbnail_url.endswith("m10000000001_1.jpg")

    auction = by_id["m10000000002"]
    assert auction.auction is not None
    assert auction.auction.total_bids == 5
    assert auction.auction.highest_bid == 22000
    expected_end = int(datetime(2026, 12, 1, 12, 0, tzinfo=UTC).timestamp())
    assert auction.auction.end_time == expected_end

    shop = by_id["shopItem001abc"]
    assert shop.is_shop_item is True
    assert shop.seller_id is None
    assert shop.url == "https://jp.mercari.com/shops/product/shopItem001abc"


def test_parse_search_empty_and_skips_bad_rows() -> None:
    empty = parse_search_response({"meta": {"numFound": "0"}, "items": []})
    assert empty.items == []
    assert empty.num_found == 0

    mixed = parse_search_response(
        {
            "meta": {"numFound": "2"},
            "items": [
                None,
                {"id": "", "name": "x", "price": "1"},
                {"id": "m1", "name": "ok", "price": "9"},
            ],
        }
    )
    assert [item.item_id for item in mixed.items] == ["m1"]

    no_auction = parse_search_response(
        {
            "items": [
                {
                    "id": "m2",
                    "name": "plain",
                    "price": "1",
                    "status": "ITEM_STATUS_ON_SALE",
                    "auction": {},
                }
            ]
        }
    )
    assert no_auction.items[0].auction is None


def test_parse_search_rejects_non_objects() -> None:
    with pytest.raises(MercariParseError):
        parse_search_response(["nope"])
    with pytest.raises(MercariParseError):
        parse_search_response({"items": "nope"})


def test_parse_item_detail_includes_seller() -> None:
    items = _package_json("items.json")
    listing = parse_item_response(items["m10000000001"])
    assert listing.item_id == "m10000000001"
    assert listing.description == "箱あり。動作確認済み。"
    assert listing.num_likes == 8
    assert listing.seller is not None
    assert listing.seller.good == 120
    assert listing.seller.star_rating == 5.0
    assert listing.condition_id == 3
    assert listing.status == "on_sale"


def test_parse_item_detail_auction() -> None:
    items = _package_json("items.json")
    listing = parse_item_response(items["m10000000002"])
    assert listing.auction is not None
    assert listing.auction.initial_price == 18000
    assert listing.auction.total_bids == 5
