from __future__ import annotations

from mercari_shopper.domain.models import Listing, SearchQuery, SearchResult


class FakeSource:
    def __init__(
        self,
        *,
        error: BaseException | None = None,
        result: SearchResult | None = None,
        items: list[Listing] | None = None,
    ) -> None:
        self.calls = 0
        self.item_calls = 0
        self.error = error
        self.result = result if result is not None else SearchResult(items=[])
        self.items = items or []
        self.queries: list[SearchQuery] = []

    async def search(self, query: SearchQuery) -> SearchResult:
        self.calls += 1
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return self.result

    async def get_items(self, item_ids: list[str]) -> list[Listing]:
        self.item_calls += 1
        if self.error is not None:
            raise self.error
        wanted = set(item_ids)
        return [item for item in self.items if item.item_id in wanted]
