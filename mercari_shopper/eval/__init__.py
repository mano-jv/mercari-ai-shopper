"""Golden-query eval harness. Offline catalog shopper plus constraint strategies."""

from mercari_shopper.eval.cases import GOLDEN_CASES, GoldenCase, validate_cases
from mercari_shopper.eval.constraints import (
    AllowedConditionsConstraint,
    Constraint,
    CountConstraint,
    PriceCapConstraint,
    PriceFloorConstraint,
)
from mercari_shopper.eval.harness import (
    CaseVerdict,
    CatalogShopper,
    CheapestRanker,
    EvalHarness,
    EvalReport,
    Ranker,
)

__all__ = [
    "GOLDEN_CASES",
    "AllowedConditionsConstraint",
    "CaseVerdict",
    "CatalogShopper",
    "CheapestRanker",
    "Constraint",
    "CountConstraint",
    "EvalHarness",
    "EvalReport",
    "GoldenCase",
    "PriceCapConstraint",
    "PriceFloorConstraint",
    "Ranker",
    "validate_cases",
]
