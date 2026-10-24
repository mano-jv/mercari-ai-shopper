from mercari_shopper.domain.models import Listing, SearchQuery, SearchResult
from mercari_shopper.sources.cache import CachedSource, TtlCache
from tests.unit.fakes import FakeSource


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_ttl_cache_expires_and_evicts() -> None:
    clock = Clock()
    cache: TtlCache[str] = TtlCache(ttl_seconds=10, max_size=2, clock=clock)
    cache.set("a", "1")
    cache.set("b", "2")
    cache.set("c", "3")
    assert cache.get("a") is None
    assert cache.get("b") == "2"
    clock.now = 11
    assert cache.get("b") is None


async def test_cached_source_returns_copy_and_skips_inner_on_hit() -> None:
    listing = Listing(item_id="m1", name="Switch", price=1000)
    inner = FakeSource(result=SearchResult(items=[listing], num_found=1), items=[listing])
    source = CachedSource(inner, TtlCache(ttl_seconds=60))
    query = SearchQuery(keyword="switch")

    first = await source.search(query)
    second = await source.search(query)
    assert inner.calls == 1
    assert first.items[0].item_id == "m1"
    first.items[0].name = "mutated"
    assert second.items[0].name == "Switch"

    details = await source.get_items(["m1"])
    again = await source.get_items(["m1"])
    assert inner.item_calls == 1
    assert details[0].price == 1000
    details[0].price = 1
    assert again[0].price == 1000
