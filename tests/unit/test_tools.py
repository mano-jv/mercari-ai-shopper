import asyncio
import json
from dataclasses import dataclass
from typing import Any

import pytest
from mercari_shopper.domain.models import (
    ItemStatus,
    Listing,
    SearchQuery,
    SearchResult,
    SortBy,
    SortOrder,
)
from mercari_shopper.logging import configure_logging
from mercari_shopper.sources.errors import MercariHttpError
from mercari_shopper.sources.fixture_source import FixtureSource
from mercari_shopper.tools.compact import ResultCompactor
from mercari_shopper.tools.errors import UnknownToolError
from mercari_shopper.tools.handlers import SubmitRecommendationsTool, Tool
from mercari_shopper.tools.registry import ToolRegistry
from mercari_shopper.tools.schemas import (
    ASK_TOOL,
    MARKET_TOOL,
    SEARCH_TOOL,
    SUBMIT_TOOL,
    GetMarketPriceArgs,
    SearchMercariArgs,
    SubmitRecommendationsArgs,
    publish_schema,
)
from tests.unit.fakes import FakeSource
from tests.unit.test_logging import log_events


@dataclass
class Call:
    id: str
    name: str
    arguments: dict[str, Any]


def _listing(
    item_id: str, name: str, *, price: int = 1000, thumb: str | None = "http://img"
) -> Listing:
    return Listing(item_id=item_id, name=name, price=price, thumbnail_url=thumb)


def test_published_schemas_are_inlined_objects() -> None:
    search = publish_schema(SearchMercariArgs)
    assert search["type"] == "object"
    assert search["additionalProperties"] is False
    assert "keyword" in search["required"]
    assert "price_min" not in search["required"]
    sort_schema = json.dumps(search["properties"]["sort"])
    for name in ("score", "newest", "price_asc", "price_desc", "likes"):
        assert name in sort_schema

    submit = publish_schema(SubmitRecommendationsArgs)
    encoded = json.dumps(submit)
    assert "$ref" not in encoded
    assert "item_id" in encoded
    assert submit["properties"]["recommendations"]["minItems"] == 3
    assert submit["properties"]["recommendations"]["maxItems"] == 3


def test_compactor_drops_junk_photoless_and_duplicates() -> None:
    compactor = ResultCompactor(description_limit=8)
    long_name = "good switch"
    listings = [
        _listing("a", "ジャンク Switch"),
        _listing("b", "plain", thumb=None),
        _listing("c", "zero", price=0),
        _listing("d", long_name, price=2500),
        _listing("d", "duplicate"),
        Listing(
            item_id="e",
            name="described",
            price=3000,
            thumbnail_url="http://img",
            description="123456789",
        ),
    ]
    compacted = compactor.compact_search(
        SearchResult(items=listings, num_found=9),
        keyword="switch",
        limit=10,
    )
    ids = [item["item_id"] for item in compacted["items"]]
    assert ids == ["d", "e"]
    assert compacted["returned"] == 2
    assert compacted["omitted"] == 4
    assert compacted["items"][1]["description"] == "1234567…"
    assert "thumbnails" not in json.dumps(compacted)

    kept = compactor.compact_search(
        SearchResult(items=[_listing("a", "ジャンク Switch")]),
        keyword="switch ジャンク",
        limit=5,
    )
    assert kept["items"][0]["item_id"] == "a"

    recovered = compactor.compact_search(
        SearchResult(
            items=[
                _listing("z", "ジャンク品", thumb=None),
                _listing("z", "good copy"),
            ]
        ),
        keyword="switch",
        limit=5,
    )
    assert [item["name"] for item in recovered["items"]] == ["good copy"]


async def test_search_handler_maps_filters_onto_the_source() -> None:
    source = FakeSource(result=SearchResult(items=[_listing("m1", "Switch")]))
    registry = ToolRegistry.for_source(source)
    execution = await registry.execute(
        Call(
            "c1",
            SEARCH_TOOL,
            {
                "keyword": " switch ",
                "price_min": 1000,
                "price_max": 30000,
                "conditions": [2, 3],
                "exclude_keywords": ["ジャンク", ""],
                "sort": "price_asc",
                "on_sale_only": False,
                "limit": 5,
            },
        )
    )
    assert execution.is_error is False
    query = source.queries[0]
    assert query == SearchQuery(
        keyword="switch",
        exclude_keyword="ジャンク",
        price_min=1000,
        price_max=30000,
        conditions=[2, 3],
        sort=SortBy.PRICE,
        order=SortOrder.ASC,
        status=[ItemStatus.ON_SALE, ItemStatus.TRADING, ItemStatus.SOLD_OUT],
        page_size=15,
    )
    payload = json.loads(execution.content)
    assert payload["items"][0]["item_id"] == "m1"
    assert payload["items"][0]["currency"] == "JPY"
    assert payload["query"]["sort"] == "price_asc"
    assert " " not in execution.content
    assert execution.elapsed_s >= 0


async def test_tool_call_log_includes_elapsed_s(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(verbose=True)

    class Slow(FakeSource):
        async def search(self, query: SearchQuery) -> SearchResult:
            await asyncio.sleep(0.05)
            return await super().search(query)

    registry = ToolRegistry.for_source(Slow(result=SearchResult(items=[_listing("m1", "Switch")])))
    execution = await registry.execute(Call("c1", SEARCH_TOOL, {"keyword": "switch"}))
    events = [row for row in log_events(capsys.readouterr().err) if row["event"] == "tool_call"]
    assert execution.is_error is False
    assert execution.elapsed_s >= 0.04
    assert events
    assert events[0]["tool"] == SEARCH_TOOL
    assert events[0]["error"] is False
    assert events[0]["elapsed_s"] >= 0.04


async def test_tool_call_error_log_includes_elapsed_s(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(verbose=True)
    registry = ToolRegistry.for_source(FakeSource())
    execution = await registry.execute(Call("c", "buy_item", {}))
    events = [row for row in log_events(capsys.readouterr().err) if row["event"] == "tool_call"]
    assert execution.is_error is True
    assert events[0]["error"] is True
    assert events[0]["elapsed_s"] >= 0


async def test_fixture_search_compacts_recorded_listings() -> None:
    registry = ToolRegistry.for_source(FixtureSource.from_package())
    execution = await registry.execute(Call("c", SEARCH_TOOL, {"keyword": "switch", "limit": 10}))
    payload = json.loads(execution.content)
    assert payload["returned"] == 3
    auction = next(item for item in payload["items"] if item["item_id"] == "m10000000002")
    shop = next(item for item in payload["items"] if item["item_id"] == "shopItem001abc")
    assert auction["auction"]["total_bids"] == 5
    assert shop["is_shop_item"] is True
    assert "thumbnails" not in execution.content


async def test_detail_handler_reports_seller_and_missing_ids() -> None:
    registry = ToolRegistry.for_source(FixtureSource.from_package())
    execution = await registry.execute(
        Call("c", "get_item_details", {"item_ids": ["m10000000001", "missing", "m10000000001"]})
    )
    payload = json.loads(execution.content)
    assert payload["missing_ids"] == ["missing"]
    item = payload["items"][0]
    assert item["seller"]["star_rating"] == 5
    assert item["num_likes"] == 8
    assert item["description"] == "箱あり。動作確認済み。"


async def test_invalid_arguments_become_an_error_result() -> None:
    registry = ToolRegistry.for_source(FakeSource())
    cases = [
        {"keyword": "switch", "price_min": 5000, "price_max": 1000},
        {"keyword": "switch", "conditions": [9]},
        {"keyword": "   "},
        {"keyword": "switch", "extra": True},
        {"item_ids": [f"m{i}" for i in range(8)]},
    ]
    names = [SEARCH_TOOL, SEARCH_TOOL, SEARCH_TOOL, SEARCH_TOOL, "get_item_details"]
    for name, arguments in zip(names, cases, strict=True):
        execution = await registry.execute(Call("c", name, arguments))
        assert execution.is_error is True
        assert execution.terminal is False
        assert json.loads(execution.content)["error"] == "invalid arguments"


async def test_unknown_tool_lists_known_names() -> None:
    registry = ToolRegistry.for_source(FakeSource())
    execution = await registry.execute(Call("c", "buy_item", {}))
    body = json.loads(execution.content)
    assert execution.is_error is True
    assert body["error"] == "unknown tool"
    assert SEARCH_TOOL in body["known_tools"]
    assert SUBMIT_TOOL in body["known_tools"]
    with pytest.raises(UnknownToolError, match="buy_item"):
        registry.get("buy_item")


async def test_source_error_is_returned_not_raised() -> None:
    source = FakeSource(error=MercariHttpError(503, "down"))
    registry = ToolRegistry.for_source(source)
    execution = await registry.execute(Call("c", SEARCH_TOOL, {"keyword": "switch"}))
    assert execution.is_error is True
    assert "503" in execution.content


async def test_submit_requires_three_unique_ranks_and_is_terminal() -> None:
    registry = ToolRegistry.for_source(FakeSource())
    bad = await registry.execute(
        Call(
            "c",
            SUBMIT_TOOL,
            {
                "summary": "two is not enough",
                "recommendations": [
                    {"item_id": "a", "rank": 1, "reason": "one"},
                    {"item_id": "b", "rank": 2, "reason": "two"},
                ],
            },
        )
    )
    assert bad.is_error is True
    assert bad.terminal is False

    good = await registry.execute(
        Call(
            "c",
            SUBMIT_TOOL,
            {
                "summary": " three picks ",
                "recommendations": [
                    {"item_id": "a", "rank": 1, "reason": "cheapest fit"},
                    {"item_id": "b", "rank": 3, "reason": "backup"},
                    {"item_id": "c", "rank": 2, "reason": "better seller"},
                ],
            },
        )
    )
    assert good.is_error is False
    assert good.terminal is True
    payload = json.loads(good.content)
    assert payload["summary"] == "three picks"
    assert [item["rank"] for item in payload["recommendations"]] == [1, 3, 2]


async def test_execute_many_isolates_failures() -> None:
    class Flaky:
        def __init__(self) -> None:
            self.calls = 0

        async def search(self, query: SearchQuery) -> SearchResult:
            del query
            self.calls += 1
            if self.calls == 1:
                raise MercariHttpError(503, "down")
            return SearchResult(items=[_listing("ok", "Switch")])

        async def get_items(self, item_ids: list[str]) -> list[Listing]:
            del item_ids
            return []

    registry = ToolRegistry.for_source(Flaky())
    results = await registry.execute_many(
        [
            Call("1", SEARCH_TOOL, {"keyword": "one"}),
            Call("2", SEARCH_TOOL, {"keyword": "two"}),
        ]
    )
    assert results[0].is_error is True
    assert results[1].is_error is False
    assert json.loads(results[1].content)["items"][0]["item_id"] == "ok"


def test_submit_and_ask_user_are_terminal() -> None:
    registry = ToolRegistry.for_source(FakeSource(), allow_clarify=True)
    assert registry.is_terminal(SUBMIT_TOOL) is True
    assert registry.is_terminal(ASK_TOOL) is True
    assert registry.is_clarify(ASK_TOOL) is True
    assert registry.is_clarify(SUBMIT_TOOL) is False
    assert registry.is_terminal(SEARCH_TOOL) is False
    assert registry.is_terminal(MARKET_TOOL) is False
    assert registry.is_terminal("buy_item") is False
    quiet = ToolRegistry.for_source(FakeSource())
    assert ASK_TOOL not in quiet.names()
    assert MARKET_TOOL in quiet.names()


def test_duplicate_registration_is_rejected() -> None:
    tool = ToolRegistry.for_source(FakeSource()).get(SEARCH_TOOL)
    with pytest.raises(ValueError, match="duplicate"):
        ToolRegistry([tool, tool])


@dataclass
class BrokenCall:
    id: str
    name: str
    arguments: dict[str, Any]
    argument_error: str | None = None


async def test_unparsed_arguments_are_rejected_before_the_handler() -> None:
    registry = ToolRegistry.for_source(FakeSource())
    execution = await registry.execute(
        BrokenCall("c", SEARCH_TOOL, {}, argument_error="tool arguments are not valid JSON")
    )
    body = json.loads(execution.content)
    assert body["error"] == "invalid arguments"
    assert "JSON" in body["detail"]
    assert registry.get(SEARCH_TOOL) is not None


class _ExplodingTool(Tool):
    name = "explode"
    description = "always fails"

    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def run(self, arguments: dict[str, Any]) -> object:
        del arguments
        raise RuntimeError("explode")


async def test_execute_many_isolates_unexpected_exceptions() -> None:
    registry = ToolRegistry([_ExplodingTool(), SubmitRecommendationsTool()])
    with pytest.raises(RuntimeError, match="explode"):
        await registry.execute(Call("1", "explode", {}))
    results = await registry.execute_many(
        [
            Call("1", "explode", {}),
            Call(
                "2",
                SUBMIT_TOOL,
                {
                    "summary": "ok",
                    "recommendations": [
                        {"item_id": "a", "rank": 1, "reason": "one"},
                        {"item_id": "b", "rank": 2, "reason": "two"},
                        {"item_id": "c", "rank": 3, "reason": "three"},
                    ],
                },
            ),
        ]
    )
    assert results[0].is_error is True
    assert "RuntimeError" in results[0].content
    assert results[1].terminal is True


async def test_market_price_tool_returns_quartiles_from_sold_listings() -> None:
    sold = SearchResult(
        items=[
            _listing(f"s{i}", "Switch OLED", price=price)
            for i, price in enumerate([18000, 20000, 22000, 24000, 26000, 28000, 30000], start=1)
        ]
        + [_listing("junk", "ジャンク Switch", price=4000)]
    )
    source = FakeSource(sold=sold)
    registry = ToolRegistry.for_source(source)
    schema = publish_schema(GetMarketPriceArgs)
    assert schema["type"] == "object"
    execution = await registry.execute(Call("c", MARKET_TOOL, {"keyword": "switch"}))
    payload = json.loads(execution.content)
    assert execution.is_error is False
    assert source.queries[0].status == [ItemStatus.SOLD_OUT]
    assert payload["sample_size"] == 7
    assert payload["confidence"] == "ok"
    assert payload["median"] == 24000
    thin = FakeSource(
        sold=SearchResult(items=[_listing("s1", "Switch", price=20000) for _ in range(3)])
    )
    thin_exec = await ToolRegistry.for_source(thin).execute(
        Call("c3", MARKET_TOOL, {"keyword": "switch"})
    )
    thin_payload = json.loads(thin_exec.content)
    assert thin_payload["confidence"] == "low"
    assert thin_payload["sample_size"] == 3


async def test_ask_user_tool_returns_the_question() -> None:
    registry = ToolRegistry.for_source(FakeSource(), allow_clarify=True)
    execution = await registry.execute(Call("c", ASK_TOOL, {"question": " Which model? "}))
    assert execution.is_error is False
    assert execution.terminal is True
    assert json.loads(execution.content)["question"] == "Which model?"


async def test_fixture_market_price_uses_sold_listings() -> None:
    registry = ToolRegistry.for_source(FixtureSource.from_package())
    execution = await registry.execute(Call("c", MARKET_TOOL, {"keyword": "switch"}))
    payload = json.loads(execution.content)
    assert payload["sample_size"] >= 5
    assert payload["confidence"] == "ok"


async def test_compactor_attaches_risk_flags() -> None:
    listing = Listing(
        item_id="r1",
        name="Switch 箱なし",
        price=20000,
        thumbnail_url="http://img",
        description="動作未確認",
    )
    compacted = ResultCompactor().compact_listing(listing)
    codes = {flag["code"] for flag in compacted["risk_flags"]}
    assert "no_box" in codes
    assert "untested" in codes


async def test_detail_results_follow_the_requested_id_order() -> None:
    source = FakeSource(items=[_listing("b", "Bee"), _listing("a", "Ay")])
    registry = ToolRegistry.for_source(source)
    execution = await registry.execute(Call("c", "get_item_details", {"item_ids": ["a", "b"]}))
    payload = json.loads(execution.content)
    assert [item["item_id"] for item in payload["items"]] == ["a", "b"]
