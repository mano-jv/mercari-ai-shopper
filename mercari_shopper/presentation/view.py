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
class FairValueView:
    sample_size: int
    median: int | None
    delta_pct: float | None
    confidence: str
    label: str


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
    fair_value: FairValueView | None = None
    risk_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResultView:
    summary: str
    items: tuple[RecommendationView, ...]
    steps: int
    tool_calls: int
    repairs: int
    elapsed_s: float
    trace: tuple[StepTrace, ...]
    question: str | None = None


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
            question=result.question,
        )


def format_yen(price: int | None) -> str:
    if price is None:
        return "¥?"
    return f"¥{price:,}"


def _card(rec: Recommendation, listing: dict[str, Any]) -> RecommendationView:
    condition = listing.get("condition")
    if not isinstance(condition, str) or not condition:
        raw_id = listing.get("condition_id")
        condition = condition_label(
            raw_id if isinstance(raw_id, int) and not isinstance(raw_id, bool) else None
        )
    flags = _risk_labels(listing.get("risk_flags"))
    return RecommendationView(
        rank=rec.rank,
        item_id=rec.item_id,
        name=str(listing.get("name") or rec.item_id),
        price=_as_int(listing.get("price")),
        condition=condition,
        url=str(listing.get("url") or ""),
        reason=rec.reason,
        pros=tuple(rec.pros),
        cons=_merged_cons(rec.cons, flags),
        seller=_seller(listing.get("seller")),
        is_shop_item=bool(listing.get("is_shop_item")),
        is_auction=listing.get("auction") is not None,
        fair_value=_fair_value(listing.get("fair_value")),
        risk_flags=flags,
    )


def _as_int(raw: object) -> int | None:
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str) and raw.strip().isdigit():
        return int(raw.strip())
    return None


def _fair_value(raw: object) -> FairValueView | None:
    if not isinstance(raw, dict):
        return None
    label = raw.get("label")
    if not isinstance(label, str) or not label.strip():
        return None
    sample = _as_int(raw.get("sample_size")) or 0
    delta = raw.get("delta_pct")
    pct = None
    if isinstance(delta, int | float) and not isinstance(delta, bool):
        pct = float(delta)
    confidence = raw.get("confidence")
    return FairValueView(
        sample_size=sample,
        median=_as_int(raw.get("median")),
        delta_pct=pct,
        confidence=str(confidence) if isinstance(confidence, str) else "none",
        label=label.strip(),
    )


def _risk_labels(raw: object) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return ()
    labels: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            label = item.get("label") or item.get("code")
            if isinstance(label, str) and label.strip():
                labels.append(label.strip())
        elif isinstance(item, str) and item.strip():
            labels.append(item.strip())
    return tuple(labels)


def _merged_cons(cons: list[str], flags: tuple[str, ...]) -> tuple[str, ...]:
    seen = {item.casefold() for item in cons}
    extra = [flag for flag in flags if flag.casefold() not in seen]
    return tuple([*cons, *extra])


def _seller(raw: object) -> SellerView | None:
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name") or "")
    stars = raw.get("star_rating")
    ratings = raw.get("num_ratings")
    rating_count = _as_int(ratings)
    star = None
    if isinstance(stars, int | float) and not isinstance(stars, bool):
        star = float(stars)
    return SellerView(name=name, star_rating=star, num_ratings=rating_count)
