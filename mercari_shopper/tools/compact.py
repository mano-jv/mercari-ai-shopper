"""Shrink listings before they are shown to the model.

Search output skips listings with no photo, zero-yen prices, and obvious junk
unless the keyword asked for junk. A second copy of an item id is skipped,
unless the first copy was dropped and a later copy is usable. Descriptions
are truncated. The model sees at most MAX_SEARCH_LIMIT rows.
"""

from __future__ import annotations

from typing import Any

from mercari_shopper.domain.models import Listing, SearchResult, condition_label
from mercari_shopper.domain.risk import (
    RiskDetector,
    default_risk_detector,
    flags_payload,
    keyword_allows_junk,
    looks_like_junk,
)
from mercari_shopper.tools.schemas import MAX_SEARCH_LIMIT

_MIN_PRICE = 1
_DEFAULT_DESCRIPTION_LIMIT = 180


class ResultCompactor:
    def __init__(
        self,
        *,
        max_items: int = MAX_SEARCH_LIMIT,
        description_limit: int = _DEFAULT_DESCRIPTION_LIMIT,
        detector: RiskDetector | None = None,
    ) -> None:
        self._max_items = max_items
        self._description_limit = description_limit
        self._detector = detector or default_risk_detector()

    def compact_search(self, result: SearchResult, *, keyword: str, limit: int) -> dict[str, Any]:
        cap = min(max(limit, 0), self._max_items)
        allow_junk = keyword_allows_junk(keyword)
        selected: list[Listing] = []
        seen: set[str] = set()
        for listing in result.items:
            if cap == 0 or len(selected) >= cap:
                break
            if listing.item_id in seen:
                continue
            if not self._keep(listing, allow_junk=allow_junk):
                continue
            seen.add(listing.item_id)
            selected.append(listing)
        return {
            "num_found": result.num_found,
            "returned": len(selected),
            "omitted": len(result.items) - len(selected),
            "items": [self.compact_listing(item) for item in selected],
        }

    def compact_details(self, listings: list[Listing], missing_ids: list[str]) -> dict[str, Any]:
        return {
            "items": [self.compact_listing(item) for item in listings],
            "missing_ids": missing_ids,
        }

    def compact_listing(self, listing: Listing) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "item_id": listing.item_id,
            "name": listing.name,
            "price": listing.price,
            "currency": "JPY",
            "status": listing.status,
            "condition_id": listing.condition_id,
            "condition": condition_label(listing.condition_id),
            "url": listing.url,
            "has_photo": bool(listing.thumbnail_url),
            "is_shop_item": listing.is_shop_item,
        }
        if listing.seller_id:
            payload["seller_id"] = listing.seller_id
        if listing.num_likes is not None:
            payload["num_likes"] = listing.num_likes
        if listing.description:
            payload["description"] = truncate(listing.description, self._description_limit)
        if listing.seller is not None:
            payload["seller"] = {
                "name": listing.seller.name,
                "star_rating": listing.seller.star_rating,
                "num_ratings": listing.seller.num_ratings,
                "good": listing.seller.good,
                "normal": listing.seller.normal,
                "bad": listing.seller.bad,
            }
        if listing.auction is not None:
            payload["auction"] = {
                "total_bids": listing.auction.total_bids,
                "highest_bid": listing.auction.highest_bid,
            }
        flags = self._detector.detect(listing)
        if flags:
            payload["risk_flags"] = flags_payload(flags)
        return payload

    def _keep(self, listing: Listing, *, allow_junk: bool) -> bool:
        if listing.price < _MIN_PRICE:
            return False
        if not listing.thumbnail_url:
            return False
        return allow_junk or not looks_like_junk(listing.name)


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit <= 1:
        return text[:limit]
    return text[: limit - 1] + "…"
