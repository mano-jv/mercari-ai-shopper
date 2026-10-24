import pytest
from mercari_shopper.domain.models import Listing, SearchQuery, SearchResult
from mercari_shopper.logging import configure_logging
from mercari_shopper.sources.auto_source import AutoSource
from mercari_shopper.sources.errors import MercariHttpError
from tests.unit.fakes import FakeSource
from tests.unit.test_logging import log_events

HIT = SearchResult(items=[], num_found=0)


@pytest.mark.parametrize("status", [401, 403, 429])
async def test_auto_source_falls_back_and_sticks(status: int) -> None:
    primary = FakeSource(error=MercariHttpError(status, "blocked"))
    fallback = FakeSource(result=HIT)
    source = AutoSource(primary, fallback)
    query = SearchQuery(keyword="switch")

    first = await source.search(query)
    second = await source.search(query)

    assert first is HIT
    assert second is HIT
    assert primary.calls == 1
    assert fallback.calls == 2


async def test_auto_source_logs_source_fallback(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(verbose=True)
    primary = FakeSource(error=MercariHttpError(403, "blocked"))
    fallback = FakeSource(result=HIT)
    await AutoSource(primary, fallback).search(SearchQuery(keyword="switch"))
    events = [
        row for row in log_events(capsys.readouterr().err) if row["event"] == "source_fallback"
    ]
    assert events == [{"event": "source_fallback", "status": 403, "method": "search"}]


async def test_auto_source_does_not_fallback_on_5xx() -> None:
    primary = FakeSource(error=MercariHttpError(503, "down"))
    fallback = FakeSource(result=HIT)
    source = AutoSource(primary, fallback)
    with pytest.raises(MercariHttpError) as raised:
        await source.search(SearchQuery(keyword="switch"))
    assert raised.value.status == 503
    assert fallback.calls == 0


async def test_auto_source_without_fallback_reraises() -> None:
    primary = FakeSource(error=MercariHttpError(401, "blocked"))
    source = AutoSource(primary, None)
    with pytest.raises(MercariHttpError) as raised:
        await source.search(SearchQuery(keyword="switch"))
    assert raised.value.status == 401


async def test_auto_source_get_items_falls_back_on_403() -> None:
    listing = Listing(item_id="m1", name="Switch", price=1000)
    primary = FakeSource(error=MercariHttpError(403, "blocked"))
    fallback = FakeSource(items=[listing])
    source = AutoSource(primary, fallback)
    details = await source.get_items(["m1"])
    assert [item.item_id for item in details] == ["m1"]
    assert primary.item_calls == 1
    assert fallback.item_calls == 1
