"""Recorded Mercari JSON. Used by tests and `--source fixture` offline demos."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from mercari_shopper.domain.models import ItemStatus, Listing, SearchQuery, SearchResult
from mercari_shopper.sources.base import SourceLifecycle, unique_ids
from mercari_shopper.sources.parser import parse_item_response, parse_search_response

_PACKAGE_FIXTURES = "mercari_shopper.fixtures"


_STATUS_VALUE = {
    ItemStatus.ON_SALE: "on_sale",
    ItemStatus.SOLD_OUT: "sold_out",
    ItemStatus.TRADING: "trading",
}


class FixtureSource(SourceLifecycle):
    def __init__(
        self,
        search_payload: dict[str, Any] | None = None,
        item_payloads: dict[str, dict[str, Any]] | None = None,
        sold_payload: dict[str, Any] | None = None,
    ) -> None:
        self._search_payload = search_payload or {"meta": {"numFound": "0"}, "items": []}
        self._item_payloads = item_payloads or {}
        self._sold_payload = sold_payload

    @classmethod
    def from_package(cls) -> FixtureSource:
        search = _load_package_json("search.json")
        items_file = _load_package_json("items.json")
        item_payloads = {
            item_id: payload for item_id, payload in items_file.items() if isinstance(payload, dict)
        }
        sold = _load_package_json("sold.json") if _package_has("sold.json") else None
        return cls(search_payload=search, item_payloads=item_payloads, sold_payload=sold)

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
        sold_path = directory / "sold.json"
        sold = json.loads(sold_path.read_text(encoding="utf-8")) if sold_path.exists() else None
        return cls(search_payload=search, item_payloads=item_payloads, sold_payload=sold)

    async def search(self, query: SearchQuery) -> SearchResult:
        items = list(_payload_items(self._search_payload))
        wants_sold = ItemStatus.SOLD_OUT in query.status
        wants_live = ItemStatus.ON_SALE in query.status or ItemStatus.TRADING in query.status
        if wants_sold and self._sold_payload is not None:
            sold_items = list(_payload_items(self._sold_payload))
            items = items + sold_items if wants_live else sold_items
        items = _match_keyword(items, query.keyword)
        allowed_status = {_STATUS_VALUE[status] for status in query.status}
        if allowed_status:
            items = [item for item in items if item.status in allowed_status]
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


def _match_keyword(items: list[Listing], keyword: str) -> list[Listing]:
    """Prefer a full-phrase hit; otherwise keep rows that contain any token.

    Offline fixtures only have a few names. A Japanese query such as
    'スイッチ 有機ELモデル' would otherwise return nothing and the agent
    would spend the whole step budget searching.
    """
    folded = keyword.casefold().strip()
    if not folded:
        return items
    exact = [item for item in items if folded in item.name.casefold()]
    if exact:
        return exact
    tokens = [token for token in folded.split() if len(token) >= 2]
    if not tokens:
        return []
    return [item for item in items if _tokens_hit(item.name, tokens)]


def _tokens_hit(name: str, tokens: list[str]) -> bool:
    folded_name = name.casefold()
    parts = [part for part in folded_name.split() if len(part) >= 2]
    for token in tokens:
        if token in folded_name:
            return True
        if any(part in token for part in parts):
            return True
    return False


def _payload_items(payload: dict[str, Any]) -> list[Listing]:
    return parse_search_response(payload).items


def _package_has(name: str) -> bool:
    return files(_PACKAGE_FIXTURES).joinpath(name).is_file()


def _load_package_json(name: str) -> dict[str, Any]:
    raw = files(_PACKAGE_FIXTURES).joinpath(name).read_text(encoding="utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"fixture {name} is not a JSON object")
    return payload
