"""Shopping types shared by sources, tools, and the agent.

Mercari condition ids, as published by the web client:

1 new / unused, 2 like new, 3 no noticeable scratches,
4 some scratches, 5 scratches and dirt, 6 overall poor.
"""

from __future__ import annotations

import json
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator

CONDITION_LABELS: dict[int, str] = {
    1: "new / unused",
    2: "like new",
    3: "no noticeable scratches",
    4: "some scratches or dirt",
    5: "scratches and dirt",
    6: "overall poor",
}


def condition_label(condition_id: int | None) -> str:
    if condition_id is None:
        return "unknown"
    return CONDITION_LABELS.get(condition_id, "unknown")


def item_url(item_id: str, is_shop_item: bool) -> str:
    if is_shop_item:
        return f"https://jp.mercari.com/shops/product/{item_id}"
    return f"https://jp.mercari.com/item/{item_id}"


class ItemStatus(StrEnum):
    ON_SALE = "STATUS_ON_SALE"
    SOLD_OUT = "STATUS_SOLD_OUT"
    TRADING = "STATUS_TRADING"


class SortBy(StrEnum):
    SCORE = "SORT_SCORE"
    CREATED_TIME = "SORT_CREATED_TIME"
    PRICE = "SORT_PRICE"
    NUM_LIKES = "SORT_NUM_LIKES"


class SortOrder(StrEnum):
    DESC = "ORDER_DESC"
    ASC = "ORDER_ASC"


def _default_status() -> list[ItemStatus]:
    return [ItemStatus.ON_SALE]


class SearchQuery(BaseModel):
    """One Mercari search. Price bounds are yen; None means no bound."""

    keyword: str = ""
    exclude_keyword: str = ""
    price_min: int | None = Field(default=None, ge=0)
    price_max: int | None = Field(default=None, ge=0)
    conditions: list[int] = Field(default_factory=list)
    sort: SortBy = SortBy.SCORE
    order: SortOrder = SortOrder.DESC
    status: list[ItemStatus] = Field(default_factory=_default_status)
    page_size: int = Field(default=40, ge=1, le=120)
    page_token: str = ""

    @field_validator("conditions")
    @classmethod
    def _check_conditions(cls, value: list[int]) -> list[int]:
        for condition_id in value:
            if condition_id < 1 or condition_id > 6:
                raise ValueError(f"condition id {condition_id} is outside 1..6")
        return value

    @model_validator(mode="after")
    def _check_prices(self) -> SearchQuery:
        if (
            self.price_min is not None
            and self.price_max is not None
            and self.price_min > self.price_max
        ):
            raise ValueError("price_min cannot exceed price_max")
        return self

    def cache_key(self) -> str:
        payload = self.model_dump(mode="json")
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)


class SellerInfo(BaseModel):
    seller_id: str
    name: str = ""
    good: int = 0
    normal: int = 0
    bad: int = 0
    num_ratings: int = 0
    star_rating: float = 0.0


class AuctionInfo(BaseModel):
    id: str = ""
    end_time: int = 0
    total_bids: int = 0
    highest_bid: int = 0
    initial_price: int = 0


class Listing(BaseModel):
    item_id: str
    name: str
    price: int
    status: str = "on_sale"
    condition_id: int | None = None
    thumbnail_url: str | None = None
    url: str = ""
    seller_id: str | None = None
    is_shop_item: bool = False
    auction: AuctionInfo | None = None
    num_likes: int | None = None
    description: str | None = None
    seller: SellerInfo | None = None
    created: int = 0

    def model_post_init(self, _context: object) -> None:
        if not self.url:
            self.url = item_url(self.item_id, self.is_shop_item)


class SearchResult(BaseModel):
    items: list[Listing]
    num_found: int = 0
    next_page_token: str = ""


class Recommendation(BaseModel):
    """One grounded pick. item_id must come from a tool result."""

    item_id: str
    rank: int
    reason: str
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
