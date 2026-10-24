"""Request body for POST /v2/entities:search.

The field set follows the public web client. A price of 0 means "no bound";
the API treats a missing bound that way, so None becomes 0 on the wire.
Sold searches also send STATUS_TRADING, which is what the site does.
"""

from __future__ import annotations

from urllib.parse import urlencode

from mercari_shopper.domain.models import ItemStatus, SearchQuery, SortBy, SortOrder

SEARCH_URL = "https://api.mercari.jp/v2/entities:search"
ITEM_URL = "https://api.mercari.jp/items/get"
SEARCH_PAGE_URL = "https://jp.mercari.com/search"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)

DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "X-Platform": "web",
    "Accept": "application/json",
    "Accept-Language": "ja",
    "Origin": "https://jp.mercari.com",
    "Referer": "https://jp.mercari.com/",
}


def item_request_url(item_id: str) -> str:
    return f"{ITEM_URL}?id={item_id}&include_item_attributes=true&include_auction=true"


def build_search_body(
    query: SearchQuery, *, session_id: str, device_uuid: str
) -> dict[str, object]:
    statuses = [status.value for status in query.status]
    if ItemStatus.SOLD_OUT in query.status and ItemStatus.TRADING.value not in statuses:
        statuses.append(ItemStatus.TRADING.value)
    return {
        "userId": "",
        "pageSize": query.page_size,
        "pageToken": query.page_token,
        "searchSessionId": session_id,
        "indexRouting": "INDEX_ROUTING_UNSPECIFIED",
        "thumbnailTypes": [],
        "searchCondition": {
            "keyword": query.keyword,
            "excludeKeyword": query.exclude_keyword,
            "sort": query.sort.value,
            "order": query.order.value,
            "status": statuses,
            "sizeId": [],
            "categoryId": [],
            "brandId": [],
            "sellerId": [],
            "priceMin": query.price_min or 0,
            "priceMax": query.price_max or 0,
            "itemConditionId": list(query.conditions),
            "shippingPayerId": [],
            "shippingFromArea": [],
            "shippingMethod": [],
            "colorId": [],
            "hasCoupon": False,
            "attributes": [],
            "itemTypes": [],
            "skuIds": [],
        },
        "serviceFrom": "suruga",
        "withItemBrand": True,
        "withItemSize": False,
        "withItemPromotions": False,
        "withItemSizes": False,
        "withShopname": True,
        "withAuction": True,
        "laplaceDeviceUuid": device_uuid,
    }


_PAGE_SORT = {
    SortBy.SCORE: "score",
    SortBy.CREATED_TIME: "created_time",
    SortBy.PRICE: "price",
    SortBy.NUM_LIKES: "likes",
}

_PAGE_STATUS = {
    ItemStatus.ON_SALE: "on_sale",
    ItemStatus.SOLD_OUT: "sold_out",
    ItemStatus.TRADING: "trading",
}


def build_search_page_url(query: SearchQuery) -> str:
    """Public search URL the browser client loads. Filters ride as query params."""
    params: list[tuple[str, str]] = [("keyword", query.keyword)]
    if query.exclude_keyword:
        params.append(("exclude_keyword", query.exclude_keyword))
    if query.price_min:
        params.append(("price_min", str(query.price_min)))
    if query.price_max:
        params.append(("price_max", str(query.price_max)))
    if query.conditions:
        params.append(("item_condition_id", ",".join(str(c) for c in query.conditions)))
    if query.status:
        params.append(("status", ",".join(_PAGE_STATUS[s] for s in query.status)))
    params.append(("sort", _PAGE_SORT[query.sort]))
    params.append(("order", "desc" if query.order is SortOrder.DESC else "asc"))
    return f"{SEARCH_PAGE_URL}?{urlencode(params)}"
