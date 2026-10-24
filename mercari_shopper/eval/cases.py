"""Ten golden shopping requests with price and condition constraints."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from mercari_shopper.domain.models import SearchQuery
from mercari_shopper.eval.constraints import (
    AllowedConditionsConstraint,
    Constraint,
    CountConstraint,
    PriceCapConstraint,
    PriceFloorConstraint,
)


@dataclass(frozen=True)
class GoldenCase:
    """One eval item. `keyword` is what the offline catalog shopper searches."""

    id: str
    query: str
    keyword: str
    price_min: int | None = None
    price_max: int | None = None
    conditions: tuple[int, ...] = ()
    min_results: int = 1

    def to_search_query(self) -> SearchQuery:
        return SearchQuery(
            keyword=self.keyword,
            price_min=self.price_min,
            price_max=self.price_max,
            conditions=list(self.conditions),
        )

    def constraints(self) -> tuple[Constraint, ...]:
        checks: list[Constraint] = [CountConstraint(self.min_results, maximum=3)]
        if self.price_min is not None:
            checks.append(PriceFloorConstraint(self.price_min))
        if self.price_max is not None:
            checks.append(PriceCapConstraint(self.price_max))
        if self.conditions:
            checks.append(AllowedConditionsConstraint(self.conditions))
        return tuple(checks)


def validate_cases(cases: Sequence[GoldenCase]) -> tuple[GoldenCase, ...]:
    rows = tuple(cases)
    ids = [case.id for case in rows]
    queries = [case.query for case in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("golden cases must have unique ids")
    if len(set(queries)) != len(queries):
        raise ValueError("golden cases must have unique queries")
    return rows


# Fixtures: ¥22,000 cond 4; ¥24,800 cond 3; ¥26,800 cond 2. Keyword "switch" hits all three.
GOLDEN_CASES: tuple[GoldenCase, ...] = (
    GoldenCase(
        id="en-oled-under-30k",
        query="a used Nintendo Switch OLED under 30000 yen in good condition",
        keyword="switch",
        price_max=30000,
        conditions=(2, 3, 4),
        min_results=3,
    ),
    GoldenCase(
        id="jp-oled-under-30k",
        query="中古のNintendo Switch 有機EL 3万円以下",
        keyword="switch",
        price_max=30000,
        min_results=3,
    ),
    GoldenCase(
        id="en-oled-under-27k",
        query="Nintendo Switch OLED under 27000 yen",
        keyword="switch",
        price_max=27000,
        min_results=3,
    ),
    GoldenCase(
        id="jp-oled-under-28k",
        query="有機EL Switch 本体 28000円まで",
        keyword="switch",
        price_max=28000,
        min_results=3,
    ),
    GoldenCase(
        id="en-price-band",
        query="used Nintendo Switch OLED between 20000 and 30000 yen",
        keyword="switch",
        price_min=20000,
        price_max=30000,
        min_results=3,
    ),
    GoldenCase(
        id="jp-no-scratches",
        query="Switch 有機EL 目立った傷や汚れなし",
        keyword="switch",
        conditions=(1, 2, 3),
        min_results=2,
    ),
    GoldenCase(
        id="en-like-new",
        query="like-new Nintendo Switch OLED under 30000 yen",
        keyword="switch",
        price_max=30000,
        conditions=(1, 2),
        min_results=1,
    ),
    GoldenCase(
        id="en-tight-budget",
        query="a Nintendo Switch OLED under 24000 yen",
        keyword="switch",
        price_max=24000,
        min_results=1,
    ),
    GoldenCase(
        id="jp-condition-3-or-better",
        query="Nintendo Switch 有機EL 30000円以下 状態が良いもの",
        keyword="switch",
        price_max=30000,
        conditions=(1, 2, 3),
        min_results=2,
    ),
    GoldenCase(
        id="en-avoid-junk",
        query="a Nintendo Switch OLED under 30000 yen, not junk",
        keyword="switch",
        price_max=30000,
        min_results=3,
    ),
)
