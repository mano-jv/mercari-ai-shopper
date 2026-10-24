import pytest
from mercari_shopper.agent.loop import AgentResult
from mercari_shopper.domain.models import Listing, Recommendation
from mercari_shopper.eval.cases import GOLDEN_CASES
from mercari_shopper.eval.constraints import (
    AllowedConditionsConstraint,
    CountConstraint,
    PriceCapConstraint,
    PriceFloorConstraint,
)
from mercari_shopper.eval.harness import CatalogShopper, CheapestRanker, EvalHarness
from mercari_shopper.sources.fixture_source import FixtureSource


def _result(
    *,
    item_id: str = "m1",
    price: int = 20000,
    condition_id: int = 3,
    count: int = 1,
) -> AgentResult:
    recs = tuple(
        Recommendation(item_id=f"{item_id}{index}", rank=index, reason="x")
        for index in range(1, count + 1)
    )
    listings = tuple(
        {"item_id": rec.item_id, "price": price, "condition_id": condition_id} for rec in recs
    )
    return AgentResult(
        summary="s",
        recommendations=recs,
        listings=listings,
        steps=1,
        tool_calls=1,
        repairs=0,
    )


def test_ten_golden_cases() -> None:
    assert len(GOLDEN_CASES) == 10
    ids = [case.id for case in GOLDEN_CASES]
    assert len(ids) == len(set(ids))
    assert all(case.query and case.keyword for case in GOLDEN_CASES)


def test_price_cap_fails_over_budget() -> None:
    failures = PriceCapConstraint(30000).check(_result(price=31000))
    assert failures == ("m11: ¥31000 exceeds cap ¥30000",)


def test_price_cap_passes_on_budget() -> None:
    assert PriceCapConstraint(30000).check(_result(price=30000)) == ()


def test_price_floor_and_missing_price() -> None:
    assert PriceFloorConstraint(20000).check(_result(price=15000))
    missing = AgentResult(
        summary="",
        recommendations=(Recommendation(item_id="m1", rank=1, reason="x"),),
        listings=({"item_id": "m1"},),
        steps=1,
        tool_calls=0,
        repairs=0,
    )
    assert "missing price" in PriceFloorConstraint(1).check(missing)[0]


def test_condition_constraint() -> None:
    assert AllowedConditionsConstraint((2, 3)).check(_result(condition_id=4))
    assert AllowedConditionsConstraint((2, 3)).check(_result(condition_id=3)) == ()


def test_count_constraint() -> None:
    assert CountConstraint(3).check(_result(count=1))
    assert CountConstraint(1, maximum=3).check(_result(count=3)) == ()
    with pytest.raises(ValueError):
        CountConstraint(-1)


def test_cheapest_ranker_is_stable() -> None:
    items = [
        Listing(item_id="b", name="b", price=100, thumbnail_url="http://x/b"),
        Listing(item_id="a", name="a", price=100, thumbnail_url="http://x/a"),
        Listing(item_id="c", name="c", price=50, thumbnail_url="http://x/c"),
    ]
    picked = CheapestRanker().pick(items, n=2)
    assert [item.item_id for item in picked] == ["c", "a"]


async def test_harness_catalog_shopper_passes_all_golden_cases() -> None:
    harness = EvalHarness()
    source = FixtureSource.from_package()
    report = await harness.run(CatalogShopper(source, harness.cases))
    assert report.ok
    assert report.passed == 10
    assert report.failed == 0


async def test_harness_records_raised_errors() -> None:
    class Boom:
        async def run(self, request: str) -> AgentResult:
            raise RuntimeError("nope")

    report = await EvalHarness().run(Boom())
    assert not report.ok
    assert report.verdicts[0].failures[0].startswith("raised RuntimeError")
