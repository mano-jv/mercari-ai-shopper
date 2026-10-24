"""The port every Mercari data source implements."""

from __future__ import annotations

from typing import Protocol, Self

from mercari_shopper.domain.models import Listing, SearchQuery, SearchResult


class MercariSource(Protocol):
    async def search(self, query: SearchQuery) -> SearchResult:
        """Return listings for a query. Empty is a valid result."""

    async def get_items(self, item_ids: list[str]) -> list[Listing]:
        """Return detail records, including seller ratings when the API has them."""


class SourceLifecycle:
    """Optional close/context-manager mixin. Factory sources always expose this."""

    async def aclose(self) -> None:
        return None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        await self.aclose()


def unique_ids(item_ids: list[str]) -> list[str]:
    """Keep order, drop blanks and duplicates so detail fetches are not repeated."""
    seen: set[str] = set()
    ordered: list[str] = []
    for item_id in item_ids:
        if not item_id or item_id in seen:
            continue
        seen.add(item_id)
        ordered.append(item_id)
    return ordered
