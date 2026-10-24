"""Recorded Mercari JSON. Used by tests and `--source fixture` offline demos."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from mercari_shopper.domain.models import Listing, SearchQuery, SearchResult
from mercari_shopper.sources.base import SourceLifecycle, unique_ids
from mercari_shopper.sources.parser import parse_item_response, parse_search_response

_PACKAGE_FIXTURES = "mercari_shopper.fixtures"


class FixtureSource(SourceLifecycle):
    def __init__(
        self,
        search_payload: dict[str, Any] | None = None,
        item_payloads: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._search_payload = search_payload or {"meta": {"numFound": "0"}, "items": []}
        self._item_payloads = item_payloads or {}

    @classmethod
    def from_package(cls) -> FixtureSource:
        search = _load_package_json("search.json")
        items_file = _load_package_json("items.json")
        item_payloads = {
            item_id: payload for item_id, payload in items_file.items() if isinstance(payload, dict)
        }
        return cls(search_payload=search, item_payloads=item_payloads)

    @classmethod
    def from_directory(cls, directory: Path) -> FixtureSource:
        search = json.loads((directory / "search.json").read_text(encoding="utf-8"))
        items_path = directory / "items.json"
        items_file = (
            json.loads(items_path.read_text(encoding="utf-8")) if items_path.exists() else {}
        )
        item_payloads = {
            item_id: payload for item_id, payload in items_file.items() if isinstance(payload, dict)
        }
        return cls(search_payload=search, item_payloads=item_payloads)

    async def search(self, query: SearchQuery) -> SearchResult:
        result = parse_search_response(self._search_payload)
        items = result.items
        keyword = query.keyword.casefold()
        if keyword:
            items = [item for item in items if keyword in item.name.casefold()]
        if query.price_min is not None:
            items = [item for item in items if item.price >= query.price_min]
        if query.price_max is not None:
            items = [item for item in items if item.price <= query.price_max]
        if query.conditions:
            allowed = set(query.conditions)
            items = [item for item in items if item.condition_id in allowed]
        return SearchResult(items=items, num_found=len(items), next_page_token="")

    async def get_items(self, item_ids: list[str]) -> list[Listing]:
        listings: list[Listing] = []
        for item_id in unique_ids(item_ids):
            payload = self._item_payloads.get(item_id)
            if payload is None:
                continue
            listings.append(parse_item_response(payload))
        return listings


def _load_package_json(name: str) -> dict[str, Any]:
    raw = files(_PACKAGE_FIXTURES).joinpath(name).read_text(encoding="utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"fixture {name} is not a JSON object")
    return payload
