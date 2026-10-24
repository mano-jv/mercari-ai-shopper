"""Eval assertions as Strategy objects. Each one reports failure strings."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from mercari_shopper.agent.loop import AgentResult


class Constraint(ABC):
    """One check against a grounded result. An empty tuple is a pass."""

    @abstractmethod
    def check(self, result: AgentResult) -> tuple[str, ...]:
        """Human-readable failures, if any."""


class CountConstraint(Constraint):
    def __init__(self, minimum: int, maximum: int | None = None) -> None:
        if minimum < 0:
            raise ValueError("minimum must be at least 0")
        if maximum is not None and maximum < minimum:
            raise ValueError("maximum cannot be below minimum")
        self.minimum = minimum
        self.maximum = maximum

    def check(self, result: AgentResult) -> tuple[str, ...]:
        count = len(result.recommendations)
        if count < self.minimum:
            return (f"expected at least {self.minimum} recommendation(s), got {count}",)
        if self.maximum is not None and count > self.maximum:
            return (f"expected at most {self.maximum} recommendation(s), got {count}",)
        return ()


class PriceCapConstraint(Constraint):
    def __init__(self, price_max: int) -> None:
        if price_max < 0:
            raise ValueError("price_max must be at least 0")
        self.price_max = price_max

    def check(self, result: AgentResult) -> tuple[str, ...]:
        failures: list[str] = []
        for rec, listing in _pairs(result):
            price = _price(listing)
            if price is None:
                failures.append(f"{rec.item_id}: missing price")
            elif price > self.price_max:
                failures.append(f"{rec.item_id}: ¥{price} exceeds cap ¥{self.price_max}")
        return tuple(failures)


class PriceFloorConstraint(Constraint):
    def __init__(self, price_min: int) -> None:
        if price_min < 0:
            raise ValueError("price_min must be at least 0")
        self.price_min = price_min

    def check(self, result: AgentResult) -> tuple[str, ...]:
        failures: list[str] = []
        for rec, listing in _pairs(result):
            price = _price(listing)
            if price is None:
                failures.append(f"{rec.item_id}: missing price")
            elif price < self.price_min:
                failures.append(f"{rec.item_id}: ¥{price} is below floor ¥{self.price_min}")
        return tuple(failures)


class AllowedConditionsConstraint(Constraint):
    def __init__(self, conditions: tuple[int, ...]) -> None:
        if not conditions:
            raise ValueError("conditions must not be empty")
        self.conditions = conditions

    def check(self, result: AgentResult) -> tuple[str, ...]:
        allowed = set(self.conditions)
        failures: list[str] = []
        for rec, listing in _pairs(result):
            condition_id = listing.get("condition_id") if isinstance(listing, dict) else None
            if not isinstance(condition_id, int):
                failures.append(f"{rec.item_id}: missing condition_id")
            elif condition_id not in allowed:
                failures.append(f"{rec.item_id}: condition {condition_id} not in {sorted(allowed)}")
        return tuple(failures)


def _pairs(result: AgentResult) -> list[tuple[Any, dict[str, Any]]]:
    listings = list(result.listings)
    rows: list[tuple[Any, dict[str, Any]]] = []
    for index, rec in enumerate(result.recommendations):
        listing = listings[index] if index < len(listings) else {}
        rows.append((rec, listing if isinstance(listing, dict) else {}))
    return rows


def _price(listing: dict[str, Any]) -> int | None:
    price = listing.get("price")
    return price if isinstance(price, int) else None
