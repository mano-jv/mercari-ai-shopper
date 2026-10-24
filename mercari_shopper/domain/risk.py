"""Rule-based risk flags. Phrase detectors and seller-trust checks, no I/O."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from mercari_shopper.domain.models import Listing, SellerInfo

JUNK_MARKERS = ("ジャンク", "動作未確認", "部品取り", "訳あり", "junk")

_PHRASES: tuple[tuple[str, str, str], ...] = (
    (r"ジャンク", "junk", "listed as junk (ジャンク)"),
    (r"動作未確認", "untested", "not tested (動作未確認)"),
    (r"訳あり(?!ま)", "flawed", "listed as flawed (訳あり)"),
    (r"傷あり(?!ま)", "scratched", "scratches noted (傷あり)"),
    (r"箱なし(?!で)", "no_box", "no box (箱なし)"),
    (r"部品取り", "for_parts", "for parts (部品取り)"),
    (r"\bjunk\b", "junk", "listed as junk (ジャンク)"),
)

NEW_SELLER_RATINGS = 5
LOW_STAR_RATING = 3.0
HIGH_BAD_RATIO = 0.15


@dataclass(frozen=True)
class RiskFlag:
    code: str
    label: str


class RiskDetector(Protocol):
    """Strategy: inspect a listing and return zero or more flags."""

    def detect(self, listing: Listing) -> tuple[RiskFlag, ...]:
        """Flags found on this listing. Empty means nothing stood out."""


class PhraseRiskDetector:
    """Regex scan of the title and description for known warning phrases."""

    def __init__(self, patterns: Sequence[tuple[str, str, str]] = _PHRASES) -> None:
        self._patterns = tuple(
            (re.compile(pattern, re.IGNORECASE), code, label) for pattern, code, label in patterns
        )

    def scan(self, text: str) -> tuple[RiskFlag, ...]:
        flags: list[RiskFlag] = []
        seen: set[str] = set()
        for pattern, code, label in self._patterns:
            if code in seen:
                continue
            if pattern.search(text):
                seen.add(code)
                flags.append(RiskFlag(code=code, label=label))
        return tuple(flags)

    def detect(self, listing: Listing) -> tuple[RiskFlag, ...]:
        blob = f"{listing.name}\n{listing.description or ''}"
        return self.scan(blob)


class SellerRiskDetector:
    """New, low-rated, or high-complaint sellers. Missing seller is not a flag."""

    def __init__(
        self,
        *,
        new_seller_ratings: int = NEW_SELLER_RATINGS,
        low_star_rating: float = LOW_STAR_RATING,
        high_bad_ratio: float = HIGH_BAD_RATIO,
    ) -> None:
        self._new_seller_ratings = new_seller_ratings
        self._low_star_rating = low_star_rating
        self._high_bad_ratio = high_bad_ratio

    def detect(self, listing: Listing) -> tuple[RiskFlag, ...]:
        return self.inspect(listing.seller)

    def inspect(self, seller: SellerInfo | None) -> tuple[RiskFlag, ...]:
        if seller is None:
            return ()
        flags: list[RiskFlag] = []
        if seller.num_ratings < self._new_seller_ratings:
            flags.append(RiskFlag(code="new_seller", label="new or rarely rated seller"))
        elif seller.star_rating > 0 and seller.star_rating < self._low_star_rating:
            flags.append(RiskFlag(code="low_rated_seller", label="low seller rating"))
        if (
            seller.num_ratings >= self._new_seller_ratings
            and seller.bad / seller.num_ratings >= self._high_bad_ratio
        ):
            flags.append(RiskFlag(code="poor_feedback", label="high share of bad ratings"))
        return tuple(flags)


class CompositeRiskDetector:
    """Runs several detectors and de-duplicates by code, preserving order."""

    def __init__(self, detectors: Sequence[RiskDetector]) -> None:
        self._detectors = list(detectors)

    def detect(self, listing: Listing) -> tuple[RiskFlag, ...]:
        flags: list[RiskFlag] = []
        seen: set[str] = set()
        for detector in self._detectors:
            for flag in detector.detect(listing):
                if flag.code in seen:
                    continue
                seen.add(flag.code)
                flags.append(flag)
        return tuple(flags)


def default_risk_detector() -> CompositeRiskDetector:
    return CompositeRiskDetector([PhraseRiskDetector(), SellerRiskDetector()])


def keyword_allows_junk(keyword: str) -> bool:
    folded = keyword.casefold()
    return any(marker.casefold() in folded for marker in JUNK_MARKERS)


def looks_like_junk(*parts: str) -> bool:
    folded = "\n".join(parts).casefold()
    return any(marker.casefold() in folded for marker in JUNK_MARKERS)


def flags_payload(flags: Sequence[RiskFlag]) -> list[dict[str, str]]:
    return [{"code": flag.code, "label": flag.label} for flag in flags]
