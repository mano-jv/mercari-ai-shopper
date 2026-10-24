"""Post-submit listing enrichment. Strategy steps behind one composite."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from mercari_shopper.domain.market import FairValueScorer, MarketSnapshot
from mercari_shopper.domain.models import Listing, SellerInfo
from mercari_shopper.domain.risk import RiskDetector, default_risk_detector, flags_payload
from mercari_shopper.logging import log_event
from mercari_shopper.sources.base import MercariSource, unique_ids
from mercari_shopper.tools.compact import ResultCompactor


class ListingEnricher(Protocol):
    """One pass over the grounded shortlist. Steps must not drop items."""

    async def enrich(
        self,
        listings: tuple[dict[str, Any], ...],
        *,
        snapshot: MarketSnapshot | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Return the same ids, with extra fields merged in."""


class SellerRatingEnricher:
    """Fetch seller ratings for shortlist rows that still lack them."""

    def __init__(
        self,
        source: MercariSource,
        compactor: ResultCompactor | None = None,
    ) -> None:
        self._source = source
        self._compactor = compactor or ResultCompactor()

    async def enrich(
        self,
        listings: tuple[dict[str, Any], ...],
        *,
        snapshot: MarketSnapshot | None = None,
    ) -> tuple[dict[str, Any], ...]:
        del snapshot
        missing = [
            str(row["item_id"])
            for row in listings
            if isinstance(row.get("item_id"), str)
            and row["item_id"]
            and not isinstance(row.get("seller"), dict)
        ]
        if not missing:
            return listings
        fetched = await self._source.get_items(unique_ids(missing))
        by_id = {item.item_id: self._compactor.compact_listing(item) for item in fetched}
        return tuple(_merge_row(row, by_id.get(str(row.get("item_id") or ""))) for row in listings)


class FairValueEnricher:
    """Attach a sold-price score when the market tool produced a usable snapshot."""

    def __init__(self, scorer: FairValueScorer | None = None) -> None:
        self._scorer = scorer or FairValueScorer()

    async def enrich(
        self,
        listings: tuple[dict[str, Any], ...],
        *,
        snapshot: MarketSnapshot | None = None,
    ) -> tuple[dict[str, Any], ...]:
        if snapshot is None:
            return listings
        enriched: list[dict[str, Any]] = []
        for row in listings:
            price = row.get("price")
            if not isinstance(price, int) or isinstance(price, bool):
                enriched.append(row)
                continue
            copy = dict(row)
            copy["fair_value"] = self._scorer.assess(price, snapshot).to_payload()
            enriched.append(copy)
        return tuple(enriched)


class RiskFlagEnricher:
    """Re-run detectors on compact rows so cons include flags even without details."""

    def __init__(self, detector: RiskDetector | None = None) -> None:
        self._detector = detector or default_risk_detector()

    async def enrich(
        self,
        listings: tuple[dict[str, Any], ...],
        *,
        snapshot: MarketSnapshot | None = None,
    ) -> tuple[dict[str, Any], ...]:
        del snapshot
        enriched: list[dict[str, Any]] = []
        for row in listings:
            listing = _listing_from_compact(row)
            flags = self._detector.detect(listing)
            if not flags:
                enriched.append(row)
                continue
            copy = dict(row)
            existing = copy.get("risk_flags")
            copy["risk_flags"] = _merge_flags(
                existing if isinstance(existing, list) else [],
                flags_payload(flags),
            )
            enriched.append(copy)
        return tuple(enriched)


class CompositeListingEnricher:
    """Pipeline: seller ratings, then fair-value, then risk flags."""

    def __init__(self, steps: Sequence[ListingEnricher]) -> None:
        self._steps = list(steps)

    async def enrich(
        self,
        listings: tuple[dict[str, Any], ...],
        *,
        snapshot: MarketSnapshot | None = None,
    ) -> tuple[dict[str, Any], ...]:
        current = listings
        for step in self._steps:
            try:
                current = await step.enrich(current, snapshot=snapshot)
            except Exception as exc:
                log_event(
                    "enrich_failed",
                    step=type(step).__name__,
                    error=type(exc).__name__,
                )
        return current


class NoOpEnricher:
    async def enrich(
        self,
        listings: tuple[dict[str, Any], ...],
        *,
        snapshot: MarketSnapshot | None = None,
    ) -> tuple[dict[str, Any], ...]:
        del snapshot
        return listings


def build_enricher(
    source: MercariSource,
    compactor: ResultCompactor | None = None,
) -> CompositeListingEnricher:
    shared = compactor or ResultCompactor()
    return CompositeListingEnricher(
        [
            SellerRatingEnricher(source, compactor=shared),
            FairValueEnricher(),
            RiskFlagEnricher(),
        ]
    )


def _merge_row(row: dict[str, Any], extra: dict[str, Any] | None) -> dict[str, Any]:
    if extra is None:
        return row
    merged = dict(row)
    if extra.get("seller"):
        merged["seller"] = extra["seller"]
    if extra.get("description") and not merged.get("description"):
        merged["description"] = extra["description"]
    if extra.get("num_likes") is not None and merged.get("num_likes") is None:
        merged["num_likes"] = extra["num_likes"]
    if extra.get("risk_flags"):
        existing = merged.get("risk_flags")
        merged["risk_flags"] = _merge_flags(
            existing if isinstance(existing, list) else [],
            extra["risk_flags"] if isinstance(extra["risk_flags"], list) else [],
        )
    return merged


def _merge_flags(existing: list[object], extra: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    merged: list[dict[str, str]] = []
    for item in [*existing, *extra]:
        if not isinstance(item, dict):
            continue
        code = item.get("code")
        label = item.get("label")
        if not isinstance(code, str) or not isinstance(label, str) or code in seen:
            continue
        seen.add(code)
        merged.append({"code": code, "label": label})
    return merged


def _listing_from_compact(row: dict[str, Any]) -> Listing:
    seller_raw = row.get("seller")
    seller = None
    if isinstance(seller_raw, dict):
        seller = SellerInfo(
            seller_id=str(row.get("seller_id") or seller_raw.get("name") or "unknown"),
            name=str(seller_raw.get("name") or ""),
            good=_as_count(seller_raw.get("good")),
            normal=_as_count(seller_raw.get("normal")),
            bad=_as_count(seller_raw.get("bad")),
            num_ratings=_as_count(seller_raw.get("num_ratings")),
            star_rating=_as_stars(seller_raw.get("star_rating")),
        )
    item_id = str(row.get("item_id") or "unknown")
    name = str(row.get("name") or item_id)
    price = row.get("price")
    return Listing(
        item_id=item_id,
        name=name,
        price=price if isinstance(price, int) and not isinstance(price, bool) else 0,
        description=str(row.get("description") or "") or None,
        seller=seller,
        thumbnail_url="x" if row.get("has_photo") else None,
    )


def _as_count(raw: object) -> int:
    if isinstance(raw, bool) or raw is None:
        return 0
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    return 0


def _as_stars(raw: object) -> float:
    if isinstance(raw, bool) or raw is None:
        return 0.0
    if isinstance(raw, int | float):
        return float(raw)
    return 0.0
