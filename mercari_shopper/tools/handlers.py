"""Tool strategies. Each one validates its own arguments and calls the source port."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar

from pydantic import BaseModel, ValidationError

from mercari_shopper.domain.models import ItemStatus, SearchQuery, SortBy, SortOrder
from mercari_shopper.llm.base import ToolSpec
from mercari_shopper.sources.base import MercariSource, unique_ids
from mercari_shopper.tools.compact import ResultCompactor
from mercari_shopper.tools.errors import ToolValidationError
from mercari_shopper.tools.schemas import (
    DETAIL_DESCRIPTION,
    DETAIL_TOOL,
    SEARCH_DESCRIPTION,
    SEARCH_TOOL,
    SUBMIT_DESCRIPTION,
    SUBMIT_TOOL,
    GetItemDetailsArgs,
    SearchMercariArgs,
    SearchSort,
    SubmitRecommendationsArgs,
    format_validation_error,
    publish_schema,
)

_SORTS: dict[SearchSort, tuple[SortBy, SortOrder]] = {
    SearchSort.SCORE: (SortBy.SCORE, SortOrder.DESC),
    SearchSort.NEWEST: (SortBy.CREATED_TIME, SortOrder.DESC),
    SearchSort.PRICE_ASC: (SortBy.PRICE, SortOrder.ASC),
    SearchSort.PRICE_DESC: (SortBy.PRICE, SortOrder.DESC),
    SearchSort.LIKES: (SortBy.NUM_LIKES, SortOrder.DESC),
}


class Tool(ABC):
    """One callable the registry can advertise and run."""

    name: ClassVar[str]
    description: ClassVar[str]
    terminal: ClassVar[bool] = False

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, parameters=self.parameters())

    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for the provider tool list."""

    @abstractmethod
    async def run(self, arguments: Mapping[str, Any]) -> object:
        """Validate arguments and return a JSON-serializable payload."""


class ParsedTool[T: BaseModel](Tool):
    """Template: parse with a pydantic model, then `handle` the typed args."""

    def __init__(self, args_model: type[T]) -> None:
        self._args_model = args_model

    def parameters(self) -> dict[str, Any]:
        return publish_schema(self._args_model)

    async def run(self, arguments: Mapping[str, Any]) -> object:
        try:
            parsed = self._args_model.model_validate(arguments)
        except ValidationError as exc:
            raise ToolValidationError(self.name, format_validation_error(exc)) from exc
        return await self.handle(parsed)

    @abstractmethod
    async def handle(self, args: T) -> object:
        """Run the tool with arguments that already passed the schema."""


class SearchMercariTool(ParsedTool[SearchMercariArgs]):
    name = SEARCH_TOOL
    description = SEARCH_DESCRIPTION

    def __init__(
        self,
        source: MercariSource,
        compactor: ResultCompactor | None = None,
    ) -> None:
        super().__init__(SearchMercariArgs)
        self._source = source
        self._compactor = compactor or ResultCompactor()

    async def handle(self, args: SearchMercariArgs) -> object:
        result = await self._source.search(to_search_query(args))
        compacted = self._compactor.compact_search(result, keyword=args.keyword, limit=args.limit)
        compacted["query"] = {
            "keyword": args.keyword,
            "price_min": args.price_min,
            "price_max": args.price_max,
            "conditions": list(args.conditions),
            "exclude_keywords": list(args.exclude_keywords),
            "sort": args.sort.value,
            "on_sale_only": args.on_sale_only,
            "limit": args.limit,
        }
        return compacted


class GetItemDetailsTool(ParsedTool[GetItemDetailsArgs]):
    name = DETAIL_TOOL
    description = DETAIL_DESCRIPTION

    def __init__(
        self,
        source: MercariSource,
        compactor: ResultCompactor | None = None,
    ) -> None:
        super().__init__(GetItemDetailsArgs)
        self._source = source
        self._compactor = compactor or ResultCompactor()

    async def handle(self, args: GetItemDetailsArgs) -> object:
        ordered = unique_ids(args.item_ids)
        listings = await self._source.get_items(ordered)
        by_id = {listing.item_id: listing for listing in listings}
        found = [by_id[item_id] for item_id in ordered if item_id in by_id]
        missing = [item_id for item_id in ordered if item_id not in by_id]
        return self._compactor.compact_details(found, missing)


class SubmitRecommendationsTool(ParsedTool[SubmitRecommendationsArgs]):
    """Terminal tool. The agent loop stops when this call succeeds."""

    name = SUBMIT_TOOL
    description = SUBMIT_DESCRIPTION
    terminal = True

    def __init__(self) -> None:
        super().__init__(SubmitRecommendationsArgs)

    async def handle(self, args: SubmitRecommendationsArgs) -> object:
        return {
            "summary": args.summary,
            "recommendations": [item.model_dump() for item in args.recommendations],
        }


# Ask Mercari for more rows than `limit`. Compaction drops junk and photoless
# listings, and the model should still receive a full page when those exist.
_PAGE_HEADROOM = 3


def to_search_query(args: SearchMercariArgs) -> SearchQuery:
    sort, order = _SORTS[args.sort]
    status = (
        [ItemStatus.ON_SALE]
        if args.on_sale_only
        else [ItemStatus.ON_SALE, ItemStatus.TRADING, ItemStatus.SOLD_OUT]
    )
    return SearchQuery(
        keyword=args.keyword,
        exclude_keyword=" ".join(args.exclude_keywords),
        price_min=args.price_min,
        price_max=args.price_max,
        conditions=list(args.conditions),
        sort=sort,
        order=order,
        status=status,
        page_size=min(120, args.limit * _PAGE_HEADROOM),
    )
