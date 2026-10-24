"""Tool argument models. The published JSON Schema is generated from these."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

SEARCH_TOOL = "search_mercari"
DETAIL_TOOL = "get_item_details"
MARKET_TOOL = "get_market_price"
ASK_TOOL = "ask_user"
SUBMIT_TOOL = "submit_recommendations"

MAX_SEARCH_LIMIT = 40
DEFAULT_SEARCH_LIMIT = 20
MAX_DETAIL_IDS = 7
MARKET_PAGE_SIZE = 40

SEARCH_DESCRIPTION = (
    "Search Mercari Japan. Prefer Japanese keywords, model numbers and exclusions. "
    "Prices are yen. Condition ids: 1 new, 2 like new, 3 no noticeable scratches, "
    "4 some scratches, 5 scratches and dirt, 6 poor. "
    "The result is compact: junk, photoless and zero-yen listings may be omitted."
)
DETAIL_DESCRIPTION = (
    "Fetch seller rating, likes and a short description for item ids already "
    "returned by search_mercari. At most 7 ids. Call this on the shortlist "
    "before ranking."
)
MARKET_DESCRIPTION = (
    "Look up recently sold Mercari prices for a keyword. Returns the median, "
    "quartiles and sample size. Use this to judge whether an asking price is fair. "
    "Fewer than 5 sold listings is low confidence, not a score."
)
ASK_DESCRIPTION = (
    "Ask the user one clarifying question when the request names no product, "
    "category, or budget. Call this instead of searching. At most once per run."
)
SUBMIT_DESCRIPTION = (
    "Finish with exactly three ranked recommendations. Every item_id must come "
    "from a previous tool result. Call this once, when the three picks are chosen."
)

ConditionId = Annotated[int, Field(ge=1, le=6)]


class SearchSort(StrEnum):
    SCORE = "score"
    NEWEST = "newest"
    PRICE_ASC = "price_asc"
    PRICE_DESC = "price_desc"
    LIKES = "likes"


class SearchMercariArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keyword: str = Field(
        min_length=1, max_length=200, description="Search text, preferably Japanese."
    )
    price_min: int | None = Field(default=None, ge=0, description="Minimum price in yen.")
    price_max: int | None = Field(default=None, ge=0, description="Maximum price in yen.")
    conditions: list[ConditionId] = Field(
        default_factory=list,
        description="Mercari condition ids from 1 (new) to 6 (poor).",
    )
    exclude_keywords: list[str] = Field(
        default_factory=list,
        description="Words that must not appear. Joined into one exclusion string.",
    )
    sort: SearchSort = Field(default=SearchSort.SCORE, description="How to order hits.")
    on_sale_only: bool = Field(
        default=True, description="When false, sold and trading listings are included."
    )
    limit: int = Field(
        default=DEFAULT_SEARCH_LIMIT,
        ge=1,
        le=MAX_SEARCH_LIMIT,
        description="Maximum listings to return after compaction.",
    )

    @field_validator("keyword")
    @classmethod
    def _strip_keyword(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("keyword is empty")
        return stripped

    @field_validator("exclude_keywords")
    @classmethod
    def _strip_excludes(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]

    @model_validator(mode="after")
    def _check_prices(self) -> SearchMercariArgs:
        if (
            self.price_min is not None
            and self.price_max is not None
            and self.price_min > self.price_max
        ):
            raise ValueError("price_min cannot exceed price_max")
        return self


class GetItemDetailsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_ids: list[str] = Field(
        min_length=1,
        max_length=MAX_DETAIL_IDS,
        description="Ids from search_mercari. At most 7.",
    )

    @field_validator("item_ids")
    @classmethod
    def _strip_ids(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        if any(not item for item in cleaned):
            raise ValueError("item id is empty")
        return cleaned


class GetMarketPriceArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keyword: str = Field(
        min_length=1,
        max_length=200,
        description="Same Japanese keyword used to search live listings.",
    )
    conditions: list[ConditionId] = Field(
        default_factory=list,
        description="Optional condition ids to match the live search.",
    )
    exclude_keywords: list[str] = Field(
        default_factory=list,
        description="Words that must not appear in the sold sample.",
    )

    @field_validator("keyword")
    @classmethod
    def _strip_keyword(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("keyword is empty")
        return stripped

    @field_validator("exclude_keywords")
    @classmethod
    def _strip_excludes(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]


class AskUserArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        min_length=1,
        max_length=400,
        description="One short question in the user's language.",
    )

    @field_validator("question")
    @classmethod
    def _strip_question(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question is empty")
        return stripped


class RecommendationArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(min_length=1, description="Id taken from a tool result.")
    rank: int = Field(ge=1, le=3)
    reason: str = Field(min_length=1, max_length=300)
    pros: list[str] = Field(default_factory=list, max_length=6)
    cons: list[str] = Field(default_factory=list, max_length=6)

    @field_validator("item_id", "reason")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped

    @field_validator("pros", "cons")
    @classmethod
    def _strip_points(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]


class SubmitRecommendationsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recommendations: list[RecommendationArgs] = Field(min_length=3, max_length=3)
    summary: str = Field(min_length=1, max_length=2000)

    @field_validator("summary")
    @classmethod
    def _strip_summary(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("summary is empty")
        return stripped

    @model_validator(mode="after")
    def _check_set(self) -> SubmitRecommendationsArgs:
        ranks = [item.rank for item in self.recommendations]
        if sorted(ranks) != [1, 2, 3]:
            raise ValueError("ranks must be 1, 2 and 3")
        ids = [item.item_id for item in self.recommendations]
        if len(set(ids)) != len(ids):
            raise ValueError("item_id values must be unique")
        return self


def publish_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON Schema for a provider tool, with $ref inlined and titles removed."""
    raw = model.model_json_schema()
    defs = raw.pop("$defs", {})
    if not isinstance(defs, dict):
        defs = {}
    node = _dereference(raw, defs)
    if not isinstance(node, dict):
        raise TypeError("tool schema root must be an object")
    cleaned = _strip_keys(node, frozenset({"title"}))
    if not isinstance(cleaned, dict):
        raise TypeError("tool schema root must be an object")
    cleaned.pop("description", None)
    if cleaned.get("type") == "object":
        cleaned.setdefault("additionalProperties", False)
    return cleaned


def format_validation_error(exc: ValidationError) -> str:
    parts: list[str] = []
    for err in exc.errors():
        location = ".".join(str(part) for part in err.get("loc", ()))
        message = str(err.get("msg", "invalid"))
        parts.append(f"{location}: {message}" if location else message)
    return "; ".join(parts) if parts else "invalid arguments"


def _dereference(node: object, defs: dict[str, Any], depth: int = 0) -> object:
    if depth > 8:
        return node
    if isinstance(node, list):
        return [_dereference(item, defs, depth + 1) for item in node]
    if not isinstance(node, dict):
        return node
    if "$ref" in node:
        ref = node["$ref"]
        name = ref.rsplit("/", 1)[-1] if isinstance(ref, str) else ""
        target = defs.get(name, {})
        resolved = _dereference(target, defs, depth + 1)
        extras = {
            key: _dereference(value, defs, depth + 1)
            for key, value in node.items()
            if key != "$ref"
        }
        if isinstance(resolved, dict):
            return {**resolved, **extras}
        return resolved
    return {key: _dereference(value, defs, depth + 1) for key, value in node.items()}


def _strip_keys(node: object, keys: frozenset[str]) -> object:
    if isinstance(node, list):
        return [_strip_keys(item, keys) for item in node]
    if isinstance(node, dict):
        return {key: _strip_keys(value, keys) for key, value in node.items() if key not in keys}
    return node
