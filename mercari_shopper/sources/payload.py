"""Request body for POST /v2/entities:search.

The field set follows the public web client. A price of 0 means "no bound";
the API treats a missing bound that way, so None becomes 0 on the wire.
Sold searches also send STATUS_TRADING, which is what the site does.
"""

from __future__ import annotations

from mercari_shopper.domain.models import ItemStatus, SearchQuery

SEARCH_URL = "https://api.mercari.jp/v2/entities:search"
ITEM_URL = "https://api.mercari.jp/items/get"


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
