"""Sold-price statistics and fair-value scoring. Pure domain, no I/O."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from mercari_shopper.domain.models import Listing
from mercari_shopper.domain.risk import keyword_allows_junk, looks_like_junk

MIN_SAMPLE_FOR_SCORE = 5
Confidence = Literal["ok", "low", "none"]


def quantile(sorted_values: Sequence[int], p: float) -> float:
    """Linear interpolation (Hyndman-Fan type 7, numpy's default)."""
    if not sorted_values:
        raise ValueError("quantile of an empty sample")
    if p < 0 or p > 1:
        raise ValueError("p must be in [0, 1]")
    n = len(sorted_values)
    if n == 1:
        return float(sorted_values[0])
    h = (n - 1) * p
    lo = int(h)
    hi = min(lo + 1, n - 1)
    frac = h - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def collect_sold_prices(listings: Sequence[Listing], *, keyword: str) -> tuple[int, ...]:
    """Keep positive prices; drop junk unless the keyword asked for it."""
    allow_junk = keyword_allows_junk(keyword)
    prices: list[int] = []
    for listing in listings:
        if listing.price < 1:
            continue
        if not allow_junk and looks_like_junk(listing.name, listing.description or ""):
            continue
        prices.append(listing.price)
    return tuple(prices)


@dataclass(frozen=True)
class MarketSnapshot:
    """Quartiles of sold prices for one keyword."""

    keyword: str
    sample_size: int
    median: float | None
    q1: float | None
    q3: float | None
    min_price: int | None
    max_price: int | None
    confidence: Confidence
    note: str

    @classmethod
    def from_prices(
        cls,
        prices: Sequence[int],
        *,
        keyword: str,
        min_sample: int = MIN_SAMPLE_FOR_SCORE,
    ) -> MarketSnapshot:
        if not prices:
            return cls(
                keyword=keyword,
                sample_size=0,
                median=None,
                q1=None,
                q3=None,
                min_price=None,
                max_price=None,
                confidence="none",
                note="no sold listings to score against",
            )
        ordered = tuple(sorted(prices))
        n = len(ordered)
        low = n < min_sample
        return cls(
            keyword=keyword,
            sample_size=n,
            median=quantile(ordered, 0.5),
            q1=quantile(ordered, 0.25),
            q3=quantile(ordered, 0.75),
            min_price=ordered[0],
            max_price=ordered[-1],
            confidence="low" if low else "ok",
            note=(
                f"low confidence ({n} sold listings); score withheld"
                if low
                else f"based on {n} sold listings"
            ),
        )

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> MarketSnapshot | None:
        keyword = payload.get("keyword")
        sample = payload.get("sample_size")
        if not isinstance(keyword, str) or not isinstance(sample, int):
            return None
        raw_confidence = payload.get("confidence")
        confidence: Confidence = (
            raw_confidence if raw_confidence in {"ok", "low", "none"} else "none"
        )
        median = payload.get("median")
        q1 = payload.get("q1")
        q3 = payload.get("q3")
        return cls(
            keyword=keyword,
            sample_size=sample,
            median=_as_float(median),
            q1=_as_float(q1),
            q3=_as_float(q3),
            min_price=_as_int(payload.get("min")),
            max_price=_as_int(payload.get("max")),
            confidence=confidence,
            note=str(payload.get("note") or ""),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "keyword": self.keyword,
            "sample_size": self.sample_size,
            "median": _round_yen(self.median),
            "q1": _round_yen(self.q1),
            "q3": _round_yen(self.q3),
            "min": self.min_price,
            "max": self.max_price,
            "confidence": self.confidence,
            "note": self.note,
        }


@dataclass(frozen=True)
class FairValueAssessment:
    """How a live listing compares to the sold-price snapshot."""

    sample_size: int
    median: int | None
    delta_pct: float | None
    confidence: Confidence
    label: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "sample_size": self.sample_size,
            "median": self.median,
            "delta_pct": self.delta_pct,
            "confidence": self.confidence,
            "label": self.label,
        }


class FairValueScorer:
    """Strategy: attach a percent-vs-median label, or withhold it when the sample is thin."""

    def __init__(self, min_sample: int = MIN_SAMPLE_FOR_SCORE) -> None:
        if min_sample < 1:
            raise ValueError("min_sample must be at least 1")
        self._min_sample = min_sample

    def assess(self, price: int, snapshot: MarketSnapshot) -> FairValueAssessment:
        n = snapshot.sample_size
        median = _round_yen(snapshot.median)
        if n == 0 or snapshot.confidence == "none" or median is None or median <= 0:
            return FairValueAssessment(
                sample_size=n,
                median=median,
                delta_pct=None,
                confidence="none",
                label="no sold-price sample",
            )
        if n < self._min_sample or snapshot.confidence == "low":
            return FairValueAssessment(
                sample_size=n,
                median=median,
                delta_pct=None,
                confidence="low",
                label=f"low confidence ({n} sold listings)",
            )
        delta = (price - median) / median * 100.0
        rounded = round(abs(delta))
        if delta <= -1:
            relation = f"about {rounded}% below typical sold price"
        elif delta >= 1:
            relation = f"about {rounded}% above typical sold price"
        else:
            relation = "about typical sold price"
        return FairValueAssessment(
            sample_size=n,
            median=median,
            delta_pct=round(delta, 1),
            confidence="ok",
            label=f"{relation} (based on {n} sold listings)",
        )


class SoldPriceAnalyzer:
    """Turns sold listings into a MarketSnapshot. Junk is dropped from the sample."""

    def __init__(self, min_sample: int = MIN_SAMPLE_FOR_SCORE) -> None:
        self._min_sample = min_sample

    def analyze(self, listings: Sequence[Listing], *, keyword: str) -> MarketSnapshot:
        prices = collect_sold_prices(listings, keyword=keyword)
        return MarketSnapshot.from_prices(prices, keyword=keyword, min_sample=self._min_sample)


def _round_yen(value: float | None) -> int | None:
    if value is None:
        return None
    return round(value)


def _as_float(raw: object) -> float | None:
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, int | float):
        return float(raw)
    return None


def _as_int(raw: object) -> int | None:
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, int):
        return raw
    return None
