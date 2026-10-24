"""Try the signed API first; switch to the browser on 401, 403 or 429."""

from __future__ import annotations

from mercari_shopper.domain.models import Listing, SearchQuery, SearchResult
from mercari_shopper.logging import log_event
from mercari_shopper.sources.base import MercariSource, SourceLifecycle
from mercari_shopper.sources.errors import MercariHttpError

FALLBACK_STATUSES = frozenset({401, 403, 429})


class AutoSource(SourceLifecycle):
    """Sticky failover: once the browser is chosen, keep using it for this session."""

    def __init__(self, primary: MercariSource, fallback: MercariSource | None) -> None:
        self._primary = primary
        self._fallback = fallback
        self._use_fallback = False

    async def search(self, query: SearchQuery) -> SearchResult:
        if self._use_fallback:
            return await self._require_fallback().search(query)
        try:
            return await self._primary.search(query)
        except MercariHttpError as error:
            self._maybe_switch(error, "search")
            if not self._use_fallback:
                raise
            return await self._require_fallback().search(query)

    async def get_items(self, item_ids: list[str]) -> list[Listing]:
        if self._use_fallback:
            return await self._require_fallback().get_items(item_ids)
        try:
            return await self._primary.get_items(item_ids)
        except MercariHttpError as error:
            self._maybe_switch(error, "get_items")
            if not self._use_fallback:
                raise
            return await self._require_fallback().get_items(item_ids)

    async def aclose(self) -> None:
        for source in (self._primary, self._fallback):
            close = getattr(source, "aclose", None)
            if close is not None:
                await close()

    def _maybe_switch(self, error: MercariHttpError, method: str) -> None:
        if error.status not in FALLBACK_STATUSES or self._fallback is None:
            return
        log_event("source_fallback", status=error.status, method=method)
        self._use_fallback = True

    def _require_fallback(self) -> MercariSource:
        if self._fallback is None:
            raise MercariHttpError(503, "browser fallback is not available")
        return self._fallback
