"""Integration: full loop against FixtureSource with a scripted LLM."""

import pytest
from mercari_shopper.agent.loop import ShopperAgent
from mercari_shopper.llm.base import LLMResponse, Message, Role, ToolCall
from mercari_shopper.sources.fixture_source import FixtureSource
from mercari_shopper.tools.registry import ToolRegistry
from mercari_shopper.tools.schemas import DETAIL_TOOL, SEARCH_TOOL, SUBMIT_TOOL
from tests.unit.fakes import ScriptedLLM

pytestmark = pytest.mark.integration

_FIXTURE_IDS = ("m10000000001", "m10000000002", "shopItem001abc")


def _rec(item_id: str, rank: int) -> dict[str, object]:
    return {
        "item_id": item_id,
        "rank": rank,
        "reason": "matches the OLED budget",
        "pros": ["price"],
        "cons": ["used"],
    }


def _tools(*calls: ToolCall) -> LLMResponse:
    return LLMResponse(
        message=Message(role=Role.ASSISTANT, tool_calls=calls),
        stop_reason="tool_use",
    )


async def test_full_loop_with_fixtures_and_scripted_llm() -> None:
    registry = ToolRegistry.for_source(FixtureSource.from_package())
    llm = ScriptedLLM(
        [
            _tools(
                ToolCall(
                    "s",
                    SEARCH_TOOL,
                    {"keyword": "switch", "price_max": 30000, "limit": 10},
                )
            ),
            _tools(ToolCall("d", DETAIL_TOOL, {"item_ids": ["m10000000001"]})),
            _tools(
                ToolCall(
                    "f",
                    SUBMIT_TOOL,
                    {
                        "summary": "OLED Switch options under 30000 yen",
                        "recommendations": [
                            _rec(_FIXTURE_IDS[0], 1),
                            _rec(_FIXTURE_IDS[1], 2),
                            _rec(_FIXTURE_IDS[2], 3),
                        ],
                    },
                )
            ),
        ]
    )
    result = await ShopperAgent(llm, registry).run(
        "a used Nintendo Switch OLED under 30000 yen in good condition"
    )
    assert [rec.item_id for rec in result.recommendations] == list(_FIXTURE_IDS)
    assert result.listings[0]["price"] <= 30000
    assert result.listings[0]["item_id"] == "m10000000001"
    assert result.listings[0].get("seller") is not None
    assert result.repairs == 0
    assert result.tool_calls == 2
    assert "30000" in result.summary
