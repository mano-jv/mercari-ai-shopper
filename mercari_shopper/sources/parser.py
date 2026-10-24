"""Turn Mercari search and item-detail JSON into domain models.

Search and Playwright interception return the same `entities:search` body.
Item detail (`items/get`) wraps the listing in a `data` object and uses
snake_case field names. Both shapes are accepted here so there is one parser.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

from mercari_shopper.domain.models import AuctionInfo, Listing, SearchResult, SellerInfo, item_url
from mercari_shopper.sources.errors import MercariParseError

_STATUS_ALIASES = {
    "ITEM_STATUS_ON_SALE": "on_sale",
    "ITEM_STATUS_SOLD_OUT": "sold_out",
    "ITEM_STATUS_TRADING": "trading",
    "STATUS_ON_SALE": "on_sale",
    "STATUS_SOLD_OUT": "sold_out",
    "STATUS_TRADING": "trading",
    "on_sale": "on_sale",
    "sold_out": "sold_out",
    "trading": "trading",
}

_SHOP_ITEM_TYPES = frozenset({"ITEM_TYPE_BEYOND", "ITEM_TYPE_MERCARI_SHOPS"})


def parse_search_response(payload: object) -> SearchResult:
    body = _as_object(payload, "search response")
    raw_items = body.get("items", [])
    if raw_items is None:
        raw_items = []
    if not isinstance(raw_items, list):
        raise MercariParseError("search items is not a list")
    listings = [item for raw in raw_items if (item := _parse_search_item(raw)) is not None]
    meta = body.get("meta")
    meta_obj = meta if isinstance(meta, dict) else {}
    return SearchResult(
        items=listings,
        num_found=_as_int(meta_obj.get("numFound", len(listings))),
        next_page_token=_as_str(meta_obj.get("nextPageToken")),
    )


def parse_item_response(payload: object) -> Listing:
    body = _as_object(payload, "item response")
    data = body.get("data")
    if isinstance(data, dict):
        return _parse_detail_item(data)
    if "id" in body or "name" in body:
        search_item = _parse_search_item(body)
        if search_item is not None:
            return search_item
        return _parse_detail_item(body)
    raise MercariParseError("item response has no listing payload")


def _parse_search_item(raw: object) -> Listing | None:
    if not isinstance(raw, dict):
        return None
    item_id = _as_str(raw.get("id"))
    name = _as_str(raw.get("name"))
    if not item_id or not name:
        return None
    is_shop = _is_shop_item(raw)
    auction = _parse_search_auction(raw.get("auction"))
    seller_id = _as_str(raw.get("sellerId")) or None
    if seller_id == "0":
        seller_id = None
    return Listing(
        item_id=item_id,
        name=name,
        price=_as_int(raw.get("price")),
        status=_normalize_status(raw.get("status")),
        condition_id=_as_int_or_none(raw.get("itemConditionId")),
        thumbnail_url=_first_thumbnail(raw.get("thumbnails")),
        url=item_url(item_id, is_shop),
        seller_id=seller_id,
        is_shop_item=is_shop,
        auction=auction,
        created=_as_int(raw.get("created")),
    )


def _parse_detail_item(raw: dict[str, Any]) -> Listing:
    item_id = _as_str(raw.get("id"))
    name = _as_str(raw.get("name"))
    if not item_id or not name:
        raise MercariParseError("item detail is missing id or name")
    is_shop = _is_shop_item(raw)
    condition = raw.get("item_condition")
    condition_id = None
    if isinstance(condition, dict):
        condition_id = _as_int_or_none(condition.get("id"))
    elif raw.get("itemConditionId") is not None:
        condition_id = _as_int_or_none(raw.get("itemConditionId"))
    seller = _parse_seller(raw.get("seller"))
    seller_id = seller.seller_id if seller is not None else (_as_str(raw.get("sellerId")) or None)
    return Listing(
        item_id=item_id,
        name=name,
        price=_as_int(raw.get("price")),
        status=_normalize_status(raw.get("status")),
        condition_id=condition_id,
        thumbnail_url=_first_thumbnail(raw.get("thumbnails")) or _first_photo(raw.get("photos")),
        url=item_url(item_id, is_shop),
        seller_id=seller_id,
        is_shop_item=is_shop,
        auction=_parse_detail_auction(raw.get("auction_info") or raw.get("auction")),
        num_likes=_as_int_or_none(raw.get("num_likes")),
        description=_as_str(raw.get("description")) or None,
        seller=seller,
        created=_as_int(raw.get("created")),
    )


def _parse_seller(raw: object) -> SellerInfo | None:
    if not isinstance(raw, dict):
        return None
    seller_id = _as_str(raw.get("id"))
    if not seller_id:
        return None
    ratings = raw.get("ratings") if isinstance(raw.get("ratings"), dict) else {}
    ratings_obj = cast(dict[str, Any], ratings)
    return SellerInfo(
        seller_id=seller_id,
        name=_as_str(raw.get("name")),
        good=_as_int(ratings_obj.get("good")),
        normal=_as_int(ratings_obj.get("normal")),
        bad=_as_int(ratings_obj.get("bad")),
        num_ratings=_as_int(raw.get("num_ratings")),
        star_rating=float(_as_int(raw.get("star_rating_score"))),
    )


def _parse_search_auction(raw: object) -> AuctionInfo | None:
    if not isinstance(raw, dict):
        return None
    if not any(raw.get(key) for key in ("id", "bidDeadline", "totalBid", "highestBid")):
        return None
    return AuctionInfo(
        id=_as_str(raw.get("id")),
        end_time=_parse_time(raw.get("bidDeadline")),
        total_bids=_as_int(raw.get("totalBid")),
        highest_bid=_as_int(raw.get("highestBid")),
    )


def _parse_detail_auction(raw: object) -> AuctionInfo | None:
    if not isinstance(raw, dict):
        return None
    return AuctionInfo(
        id=_as_str(raw.get("id")),
        end_time=_parse_time(raw.get("expected_end_time") or raw.get("bidDeadline")),
        total_bids=_as_int(raw.get("total_bids") or raw.get("totalBid")),
        highest_bid=_as_int(raw.get("highest_bid") or raw.get("highestBid")),
        initial_price=_as_int(raw.get("initial_price")),
    )


def _is_shop_item(raw: dict[str, Any]) -> bool:
    item_type = _as_str(raw.get("itemType") or raw.get("item_type"))
    if item_type in _SHOP_ITEM_TYPES:
        return True
    shop = raw.get("shop")
    if isinstance(shop, dict) and shop.get("id"):
        return True
    return _truthy(raw.get("is_shop_item"))


def _normalize_status(raw: object) -> str:
    text = _as_str(raw)
    return _STATUS_ALIASES.get(text, text.lower() or "on_sale")


def _first_thumbnail(raw: object) -> str | None:
    if isinstance(raw, list) and raw:
        first = raw[0]
        if isinstance(first, str) and first:
            return first
        if isinstance(first, dict):
            url = _as_str(first.get("uri") or first.get("url"))
            return url or None
    return None


def _first_photo(raw: object) -> str | None:
    if isinstance(raw, list) and raw:
        first = raw[0]
        if isinstance(first, str) and first:
            return first
        if isinstance(first, dict):
            url = _as_str(first.get("uri") or first.get("url") or first.get("photo_url"))
            return url or None
    return None


def _as_object(payload: object, what: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise MercariParseError(f"{what} is not a JSON object")
    return payload


def _as_str(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _as_int(value: object, default: int = 0) -> int:
    parsed = _as_int_or_none(value)
    return default if parsed is None else parsed


def _as_int_or_none(value: object) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _as_str(value).lower() in {"1", "true", "yes", "y"}


def _parse_time(value: object) -> int:
    if value is None or value == "":
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value)
    if text.isdigit():
        return int(text)
    try:
        instant = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return 0
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=UTC)
    return int(instant.timestamp())
