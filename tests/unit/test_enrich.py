from mercari_shopper.agent.enrich import build_enricher
from mercari_shopper.domain.market import MarketSnapshot
from mercari_shopper.domain.models import Listing, SellerInfo
from mercari_shopper.sources.errors import MercariHttpError
from tests.unit.fakes import FakeSource


async def test_enricher_adds_seller_fair_value_and_risk_flags() -> None:
    live = Listing(
        item_id="m1",
        name="Switch 箱なし",
        price=20000,
        thumbnail_url="http://img",
    )
    detailed = Listing(
        item_id="m1",
        name="Switch 箱なし",
        price=20000,
        thumbnail_url="http://img",
        description="箱なし。",
        seller=SellerInfo(
            seller_id="9",
            name="new-seller",
            num_ratings=1,
            star_rating=5.0,
        ),
    )
    source = FakeSource(items=[detailed])
    snapshot = MarketSnapshot.from_prices(
        [18000, 20000, 22000, 24000, 26000, 28000, 30000],
        keyword="switch",
    )
    enriched = await build_enricher(source).enrich(
        ({"item_id": "m1", "name": live.name, "price": live.price},),
        snapshot=snapshot,
    )
    row = enriched[0]
    assert row["seller"]["name"] == "new-seller"
    assert row["fair_value"]["confidence"] == "ok"
    assert "typical sold price" in row["fair_value"]["label"]
    codes = {flag["code"] for flag in row["risk_flags"]}
    assert "no_box" in codes
    assert "new_seller" in codes


async def test_enricher_survives_detail_fetch_errors() -> None:
    source = FakeSource(error=MercariHttpError(503, "down"))
    snapshot = MarketSnapshot.from_prices(
        [18000, 20000, 22000, 24000, 26000],
        keyword="switch",
    )
    enriched = await build_enricher(source).enrich(
        ({"item_id": "m1", "name": "Switch", "price": 20000},),
        snapshot=snapshot,
    )
    assert enriched[0]["item_id"] == "m1"
    assert enriched[0]["fair_value"]["confidence"] == "ok"
