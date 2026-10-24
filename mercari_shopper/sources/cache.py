"""In-memory TTL cache used as a source decorator."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from time import monotonic

from mercari_shopper.domain.models import Listing, SearchQuery, SearchResult
from mercari_shopper.sources.base import MercariSource, SourceLifecycle, unique_ids


class TtlCache[T]:
    """Small LRU map. Expired entries are dropped on read."""

    def __init__(
        self,
        ttl_seconds: float = 300.0,
        max_size: int = 256,
        *,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._clock = clock
        self._data: OrderedDict[str, tuple[float, T]] = OrderedDict()

    @property
    def enabled(self) -> bool:
        return self._ttl > 0 and self._max_size > 0

    def get(self, key: str) -> T | None:
        item = self._data.get(key)
        if item is None:
            return None
        expires_at, value = item
        if self._clock() >= expires_at:
            del self._data[key]
            return None
        self._data.move_to_end(key)
        return value

    def set(self, key: str, value: T) -> None:
        if not self.enabled:
            return
        while len(self._data) >= self._max_size:
            self._data.popitem(last=False)
        self._data[key] = (self._clock() + self._ttl, value)
        self._data.move_to_end(key)

    def clear(self) -> None:
        self._data.clear()


class CachedSource(SourceLifecycle):
    """Decorator: cache search results and item details on a MercariSource."""

    def __init__(self, inner: MercariSource, cache: TtlCache[object] | None = None) -> None:
        self._inner = inner
        self._cache: TtlCache[object] = cache if cache is not None else TtlCache()

    async def search(self, query: SearchQuery) -> SearchResult:
        key = f"search:{query.cache_key()}"
        hit = self._cache.get(key)
        if isinstance(hit, SearchResult):
            return hit.model_copy(deep=True)
        result = await self._inner.search(query)
        self._cache.set(key, result)
        return result.model_copy(deep=True)

    async def get_items(self, item_ids: list[str]) -> list[Listing]:
        found: dict[str, Listing] = {}
        missing: list[str] = []
        for item_id in unique_ids(item_ids):
            hit = self._cache.get(f"item:{item_id}")
            if isinstance(hit, Listing):
                found[item_id] = hit
            else:
                missing.append(item_id)
        if missing:
            fetched = await self._inner.get_items(missing)
            for listing in fetched:
                self._cache.set(f"item:{listing.item_id}", listing)
                found[listing.item_id] = listing
        return [
            found[item_id].model_copy(deep=True)
            for item_id in unique_ids(item_ids)
            if item_id in found
        ]

    async def aclose(self) -> None:
        close = getattr(self._inner, "aclose", None)
        if close is not None:
            await close()
