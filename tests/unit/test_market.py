from mercari_shopper.domain.market import (
    FairValueScorer,
    MarketSnapshot,
    SoldPriceAnalyzer,
    collect_sold_prices,
    quantile,
)
from mercari_shopper.domain.models import Listing


def test_quantile_type7_odd_and_even() -> None:
    odd = [10, 20, 30, 40, 50]
    assert quantile(odd, 0.5) == 30
    assert quantile(odd, 0.25) == 20
    assert quantile(odd, 0.75) == 40
    even = [10, 20, 30, 40, 50, 60]
    assert quantile(even, 0.5) == 35
    assert quantile(even, 0.25) == 22.5
    assert quantile(even, 0.75) == 47.5
    assert quantile([7], 0.5) == 7


def test_small_sample_is_low_confidence_not_a_score() -> None:
    prices = [1000, 2000, 3000, 4000]
    snapshot = MarketSnapshot.from_prices(prices, keyword="switch")
    assert snapshot.sample_size == 4
    assert snapshot.confidence == "low"
    assert snapshot.median == 2500
    assessment = FairValueScorer().assess(1800, snapshot)
    assert assessment.confidence == "low"
    assert assessment.delta_pct is None
    assert "low confidence" in assessment.label


def test_empty_sample_has_no_score() -> None:
    snapshot = MarketSnapshot.from_prices([], keyword="switch")
    assert snapshot.confidence == "none"
    assessment = FairValueScorer().assess(20000, snapshot)
    assert assessment.delta_pct is None
    assert assessment.confidence == "none"


def test_fair_value_percent_vs_median() -> None:
    prices = [18000, 20000, 22000, 24000, 26000, 28000, 30000]
    snapshot = MarketSnapshot.from_prices(prices, keyword="スイッチ")
    assert snapshot.confidence == "ok"
    assert snapshot.median == 24000
    below = FairValueScorer().assess(20000, snapshot)
    assert below.delta_pct is not None
    assert below.delta_pct < 0
    assert "below typical sold price" in below.label
    assert "based on 7 sold listings" in below.label
    above = FairValueScorer().assess(28000, snapshot)
    assert above.delta_pct is not None
    assert above.delta_pct > 0
    assert "above typical sold price" in above.label


def test_sold_sample_drops_junk_unless_asked() -> None:
    listings = [
        Listing(item_id="a", name="Switch OLED", price=20000, thumbnail_url="http://img"),
        Listing(item_id="b", name="ジャンク Switch", price=5000, thumbnail_url="http://img"),
        Listing(item_id="c", name="Switch 部品取り", price=3000, thumbnail_url="http://img"),
        Listing(item_id="d", name="zero", price=0, thumbnail_url="http://img"),
    ]
    clean = collect_sold_prices(listings, keyword="switch")
    assert clean == (20000,)
    with_junk = collect_sold_prices(listings, keyword="switch ジャンク")
    assert 5000 in with_junk
    assert 20000 in with_junk


def test_analyzer_payload_round_trip() -> None:
    listings = [
        Listing(
            item_id=f"m{i}",
            name="Switch",
            price=price,
            thumbnail_url="http://img",
        )
        for i, price in enumerate([10_000, 12_000, 14_000, 16_000, 18_000], start=1)
    ]
    snapshot = SoldPriceAnalyzer().analyze(listings, keyword="switch")
    payload = snapshot.to_payload()
    restored = MarketSnapshot.from_payload(payload)
    assert restored is not None
    assert restored.sample_size == 5
    assert restored.confidence == "ok"
    assert restored.median == snapshot.median
