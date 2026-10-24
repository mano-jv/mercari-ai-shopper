from mercari_shopper.agent.loop import AgentResult
from mercari_shopper.app import LiveShopper
from mercari_shopper.config import Settings
from mercari_shopper.llm.base import LLMResponse, Message, Role, ToolCall
from mercari_shopper.tools.schemas import SEARCH_TOOL, SUBMIT_TOOL
from tests.unit.fakes import ScriptedLLM

_FIXTURE_IDS = ("m10000000001", "m10000000002", "shopItem001abc")


def _tools(*calls: ToolCall) -> LLMResponse:
    return LLMResponse(
        message=Message(role=Role.ASSISTANT, tool_calls=calls),
        stop_reason="tool_use",
    )


def _rec(item_id: str, rank: int) -> dict[str, object]:
    return {
        "item_id": item_id,
        "rank": rank,
        "reason": "matches the OLED budget",
        "pros": ["price"],
        "cons": ["used"],
    }


async def test_live_shopper_wires_fixture_source_and_scripted_llm() -> None:
    llm = ScriptedLLM(
        [
            _tools(
                ToolCall("s", SEARCH_TOOL, {"keyword": "switch", "price_max": 30000, "limit": 10})
            ),
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
    settings = Settings(mercari_source="fixture", anthropic_api_key="test")
    async with LiveShopper(settings=settings, source_kind="fixture", llm=llm) as shopper:
        result = await shopper.run("a used Nintendo Switch OLED under 30000 yen")
    assert isinstance(result, AgentResult)
    assert [rec.item_id for rec in result.recommendations] == list(_FIXTURE_IDS)
    assert result.elapsed_s >= 0
    assert result.trace
    assert "search_mercari" in result.trace[0].tool_names
