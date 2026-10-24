import json
from pathlib import Path

from mercari_shopper.domain.models import ItemStatus, SearchQuery
from mercari_shopper.sources.fixture_source import FixtureSource


async def test_package_fixtures_filter_by_keyword_and_price() -> None:
    source = FixtureSource.from_package()
    all_hits = await source.search(SearchQuery(keyword="switch"))
    assert len(all_hits.items) == 3
    cheap = await source.search(SearchQuery(keyword="switch", price_max=25000))
    assert [item.item_id for item in cheap.items] == ["m10000000001", "m10000000002"]
    details = await source.get_items(["m10000000001", "nope"])
    assert len(details) == 1
    assert details[0].seller is not None
    tokens = await source.search(SearchQuery(keyword="スイッチ 有機ELモデル 中古"))
    assert {item.item_id for item in tokens.items} >= {"m10000000001"}


async def test_package_sold_fixture_is_separate_from_on_sale() -> None:
    source = FixtureSource.from_package()
    live = await source.search(SearchQuery(keyword="switch", status=[ItemStatus.ON_SALE]))
    sold = await source.search(SearchQuery(keyword="switch", status=[ItemStatus.SOLD_OUT]))
    assert len(live.items) == 3
    assert all(item.status == "on_sale" for item in live.items)
    assert len(sold.items) >= 5
    assert all(item.status == "sold_out" for item in sold.items)


async def test_from_directory_empty_search(tmp_path: Path) -> None:
    (tmp_path / "search.json").write_text(
        json.dumps({"meta": {"numFound": "0"}, "items": []}),
        encoding="utf-8",
    )
    source = FixtureSource.from_directory(tmp_path)
    result = await source.search(SearchQuery(keyword="anything"))
    assert result.items == []
