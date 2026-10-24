"""The port every Mercari data source implements."""

from __future__ import annotations

from typing import Protocol

from mercari_shopper.domain.models import Listing, SearchQuery, SearchResult


class MercariSource(Protocol):
    async def search(self, query: SearchQuery) -> SearchResult:
        """Return listings for a query. Empty is a valid result."""

    async def get_items(self, item_ids: list[str]) -> list[Listing]:
        """Return detail records, including seller ratings when the API has them."""
