"""Adapter from AgentResult (domain) to a display-ready view model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mercari_shopper.agent.loop import AgentResult, StepTrace
from mercari_shopper.domain.models import Recommendation, condition_label


@dataclass(frozen=True)
class SellerView:
    name: str
    star_rating: float | None
    num_ratings: int | None


@dataclass(frozen=True)
class RecommendationView:
    rank: int
    item_id: str
    name: str
    price: int | None
    condition: str
    url: str
    reason: str
    pros: tuple[str, ...]
    cons: tuple[str, ...]
    seller: SellerView | None
    is_shop_item: bool
    is_auction: bool


@dataclass(frozen=True)
class ResultView:
    summary: str
    items: tuple[RecommendationView, ...]
    steps: int
    tool_calls: int
    repairs: int
    elapsed_s: float
    trace: tuple[StepTrace, ...]


class ResultViewBuilder:
    """Maps a grounded agent result onto presentation cards."""

    def build(self, result: AgentResult) -> ResultView:
        cards: list[RecommendationView] = []
        for rec, listing in zip(result.recommendations, result.listings, strict=False):
            cards.append(_card(rec, listing if isinstance(listing, dict) else {}))
        leftover = result.recommendations[len(cards) :]
        for rec in leftover:
            cards.append(_card(rec, {"item_id": rec.item_id}))
        return ResultView(
            summary=result.summary,
            items=tuple(cards),
            steps=result.steps,
            tool_calls=result.tool_calls,
            repairs=result.repairs,
            elapsed_s=result.elapsed_s,
            trace=result.trace,
        )


def format_yen(price: int | None) -> str:
    if price is None:
        return "¥?"
    return f"¥{price:,}"


def _card(rec: Recommendation, listing: dict[str, Any]) -> RecommendationView:
    condition = listing.get("condition")
    if not isinstance(condition, str) or not condition:
        raw_id = listing.get("condition_id")
        condition = condition_label(raw_id if isinstance(raw_id, int) else None)
    price = listing.get("price")
    return RecommendationView(
        rank=rec.rank,
        item_id=rec.item_id,
        name=str(listing.get("name") or rec.item_id),
        price=price if isinstance(price, int) else None,
        condition=condition,
        url=str(listing.get("url") or ""),
        reason=rec.reason,
        pros=tuple(rec.pros),
        cons=tuple(rec.cons),
        seller=_seller(listing.get("seller")),
        is_shop_item=bool(listing.get("is_shop_item")),
        is_auction=listing.get("auction") is not None,
    )


def _seller(raw: object) -> SellerView | None:
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name") or "")
    stars = raw.get("star_rating")
    ratings = raw.get("num_ratings")
    return SellerView(
        name=name,
        star_rating=float(stars) if isinstance(stars, int | float) else None,
        num_ratings=int(ratings) if isinstance(ratings, int) else None,
    )
