from collections.abc import Sequence
from typing import Any

import pytest
from mercari_shopper.agent.budget import AgentBudget
from mercari_shopper.agent.enrich import build_enricher
from mercari_shopper.agent.errors import (
    AgentBudgetError,
    AgentGroundingError,
    AgentInputError,
    AgentLLMError,
)
from mercari_shopper.agent.factory import build_agent
from mercari_shopper.agent.guardrails import UNTRUSTED_START, InputGuard
from mercari_shopper.agent.loop import ShopperAgent
from mercari_shopper.config import Settings, load_settings
from mercari_shopper.domain.models import Listing, SearchQuery, SearchResult, SellerInfo
from mercari_shopper.llm.base import ForcedTool, LLMResponse, Message, Role, ToolCall
from mercari_shopper.llm.errors import LLMError
from mercari_shopper.sources.errors import MercariHttpError
from mercari_shopper.tools.registry import ToolRegistry
from mercari_shopper.tools.schemas import (
    ASK_TOOL,
    DETAIL_TOOL,
    MARKET_TOOL,
    SEARCH_TOOL,
    SUBMIT_TOOL,
)
from tests.unit.fakes import FakeSource, ScriptedLLM


def _listing(item_id: str, name: str, price: int = 20000) -> Listing:
    return Listing(
        item_id=item_id,
        name=name,
        price=price,
        thumbnail_url=f"http://img/{item_id}",
        condition_id=3,
    )


def _three() -> list[Listing]:
    return [
        _listing("m1", "Switch A", 20000),
        _listing("m2", "Switch B", 22000),
        _listing("m3", "Switch C", 25000),
    ]


def _rec(item_id: str, rank: int, reason: str = "fits the budget") -> dict[str, object]:
    return {
        "item_id": item_id,
        "rank": rank,
        "reason": reason,
        "pros": ["price"],
        "cons": ["used"],
    }


def _submit(ids: Sequence[str], summary: str = "three fits") -> dict[str, object]:
    first, second, third = ids
    return {
        "summary": summary,
        "recommendations": [_rec(first, 1), _rec(second, 2), _rec(third, 3)],
    }


def _tools(*calls: ToolCall, text: str = "") -> LLMResponse:
    return LLMResponse(
        message=Message(role=Role.ASSISTANT, content=text, tool_calls=calls),
        stop_reason="tool_use",
    )


def _text(content: str) -> LLMResponse:
    return LLMResponse(
        message=Message(role=Role.ASSISTANT, content=content),
        stop_reason="end_turn",
    )


def _agent(llm: ScriptedLLM, source: FakeSource | None = None, **kwargs: Any) -> ShopperAgent:
    registry = ToolRegistry.for_source(source or FakeSource(result=SearchResult(items=_three())))
    return ShopperAgent(llm, registry, **kwargs)


async def test_happy_path_search_then_submit() -> None:
    llm = ScriptedLLM(
        [
            _tools(ToolCall("s", SEARCH_TOOL, {"keyword": "switch"})),
            _tools(ToolCall("f", SUBMIT_TOOL, _submit(("m1", "m2", "m3")))),
        ]
    )
    result = await _agent(llm).run("a used Switch under 30000 yen")
    assert [rec.item_id for rec in result.recommendations] == ["m1", "m2", "m3"]
    assert result.summary == "three fits"
    assert result.steps == 2
    assert result.tool_calls == 1
    assert result.repairs == 0
    assert result.listings[0]["name"] == "Switch A"
    assert result.listings[0]["price"] == 20000
    assert result.elapsed_s >= 0
    assert result.trace[0].tool_names == ("search_mercari",)
    assert result.trace[-1].tool_names == ("submit_recommendations",)


async def test_multi_search_refinement_runs_in_parallel() -> None:
    source = FakeSource(result=SearchResult(items=_three()))
    llm = ScriptedLLM(
        [
            _tools(
                ToolCall("s1", SEARCH_TOOL, {"keyword": "switch"}),
                ToolCall("s2", SEARCH_TOOL, {"keyword": "有機EL"}),
            ),
            _tools(ToolCall("f", SUBMIT_TOOL, _submit(("m1", "m2", "m3"), "refined"))),
        ]
    )
    result = await ShopperAgent(llm, ToolRegistry.for_source(source)).run("Switch OLED")
    assert source.calls == 2
    assert [query.keyword for query in source.queries] == ["switch", "有機EL"]
    assert result.summary == "refined"
    assert result.tool_calls == 2


async def test_hallucinated_id_triggers_one_repair() -> None:
    llm = ScriptedLLM(
        [
            _tools(ToolCall("s", SEARCH_TOOL, {"keyword": "switch"})),
            _tools(ToolCall("bad", SUBMIT_TOOL, _submit(("m1", "nope", "m3")))),
            _tools(ToolCall("ok", SUBMIT_TOOL, _submit(("m1", "m2", "m3")))),
        ]
    )
    result = await _agent(llm).run("switch")
    assert result.repairs == 1
    assert [rec.item_id for rec in result.recommendations] == ["m1", "m2", "m3"]
    choice = llm.requests[2].tool_choice
    assert isinstance(choice, ForcedTool)
    assert choice.name == SUBMIT_TOOL
    repair_result = llm.requests[2].messages[-1].tool_result
    assert repair_result is not None
    assert repair_result.is_error is True
    assert "nope" in repair_result.content
    assert UNTRUSTED_START in repair_result.content


async def test_second_ungrounded_submit_raises() -> None:
    llm = ScriptedLLM(
        [
            _tools(ToolCall("s", SEARCH_TOOL, {"keyword": "switch"})),
            _tools(ToolCall("bad1", SUBMIT_TOOL, _submit(("ghost", "m2", "m3")))),
            _tools(ToolCall("bad2", SUBMIT_TOOL, _submit(("still", "m2", "m3")))),
        ]
    )
    with pytest.raises(AgentGroundingError, match="item ids"):
        await _agent(llm).run("switch")


async def test_last_step_forces_submit_when_listings_exist() -> None:
    llm = ScriptedLLM(
        [
            _tools(ToolCall("s", SEARCH_TOOL, {"keyword": "switch"})),
            _tools(ToolCall("f", SUBMIT_TOOL, _submit(("m1", "m2", "m3")))),
        ]
    )
    result = await _agent(llm, budget=AgentBudget(max_steps=2, timeout_seconds=30)).run("switch")
    choice = llm.requests[1].tool_choice
    assert isinstance(choice, ForcedTool)
    assert choice.name == SUBMIT_TOOL
    assert [rec.item_id for rec in result.recommendations] == ["m1", "m2", "m3"]
    assert any(
        message.role is Role.USER and SUBMIT_TOOL in message.content
        for message in llm.requests[1].messages
    )


async def test_three_ids_force_submit_after_two_turns() -> None:
    llm = ScriptedLLM(
        [
            _tools(ToolCall("s", SEARCH_TOOL, {"keyword": "switch"})),
            _tools(ToolCall("d", DETAIL_TOOL, {"item_ids": ["m1", "m2"]})),
            _tools(ToolCall("f", SUBMIT_TOOL, _submit(("m1", "m2", "m3")))),
        ]
    )
    source = FakeSource(result=SearchResult(items=_three()), items=_three())
    result = await ShopperAgent(
        llm, ToolRegistry.for_source(source), budget=AgentBudget(max_steps=8, timeout_seconds=30)
    ).run("switch")
    choice = llm.requests[2].tool_choice
    assert isinstance(choice, ForcedTool)
    assert choice.name == SUBMIT_TOOL
    assert result.steps == 3


async def test_max_steps_exhaustion() -> None:
    llm = ScriptedLLM(
        [
            _tools(ToolCall("s1", SEARCH_TOOL, {"keyword": "one"})),
            _tools(ToolCall("s2", SEARCH_TOOL, {"keyword": "two"})),
            _tools(ToolCall("s3", SEARCH_TOOL, {"keyword": "three"})),
        ]
    )
    agent = _agent(llm, budget=AgentBudget(max_steps=2, timeout_seconds=30))
    with pytest.raises(AgentBudgetError, match="step"):
        await agent.run("switch")
    assert len(llm.requests) == 2


async def test_tool_exception_is_returned_and_loop_continues() -> None:
    class Flaky(FakeSource):
        async def search(self, query: SearchQuery) -> SearchResult:
            self.calls += 1
            self.queries.append(query)
            if self.calls == 1:
                raise MercariHttpError(503, "down")
            return self.result

    source = Flaky(result=SearchResult(items=_three()))
    llm = ScriptedLLM(
        [
            _tools(ToolCall("s1", SEARCH_TOOL, {"keyword": "switch"})),
            _tools(ToolCall("s2", SEARCH_TOOL, {"keyword": "switch 有機EL"})),
            _tools(ToolCall("f", SUBMIT_TOOL, _submit(("m1", "m2", "m3")))),
        ]
    )
    result = await ShopperAgent(llm, ToolRegistry.for_source(source)).run("switch")
    assert source.calls == 2
    assert result.recommendations[0].item_id == "m1"
    first_tool = llm.requests[1].messages[-1].tool_result
    assert first_tool is not None
    assert first_tool.is_error is True
    assert "503" in first_tool.content


async def test_retryable_llm_error_then_succeeds() -> None:
    sleeps: list[float] = []

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    llm = ScriptedLLM(
        [
            LLMError("busy", provider="fake", status=429, retryable=True),
            _tools(ToolCall("s", SEARCH_TOOL, {"keyword": "switch"})),
            _tools(ToolCall("f", SUBMIT_TOOL, _submit(("m1", "m2", "m3")))),
        ]
    )
    result = await _agent(llm, sleep=sleep).run("switch")
    assert result.steps == 2
    assert sleeps == [0.4]
    assert len(llm.requests) == 3


async def test_non_retryable_llm_error_fails_immediately() -> None:
    llm = ScriptedLLM([LLMError("bad key", provider="fake", status=401, retryable=False)])
    with pytest.raises(AgentLLMError, match="bad key"):
        await _agent(llm).run("switch")
    assert len(llm.requests) == 1


async def test_time_budget_exhaustion() -> None:
    class SequenceClock:
        def __init__(self, values: list[float]) -> None:
            self._values = list(values)
            self._last = values[-1]

        def __call__(self) -> float:
            if self._values:
                self._last = self._values.pop(0)
            return self._last

    llm = ScriptedLLM(
        [
            _tools(ToolCall("s1", SEARCH_TOOL, {"keyword": "one"})),
            _tools(ToolCall("s2", SEARCH_TOOL, {"keyword": "two"})),
        ]
    )
    agent = _agent(
        llm,
        budget=AgentBudget(max_steps=8, timeout_seconds=10),
        clock=SequenceClock([0.0, 0.0, 100.0]),
    )
    with pytest.raises(AgentBudgetError, match="time"):
        await agent.run("switch")
    assert len(llm.requests) == 1


async def test_tool_call_budget_blocks_extra_searches() -> None:
    llm = ScriptedLLM(
        [
            _tools(ToolCall("s1", SEARCH_TOOL, {"keyword": "one"})),
            _tools(ToolCall("s2", SEARCH_TOOL, {"keyword": "two"})),
        ]
    )
    agent = _agent(llm, budget=AgentBudget(max_steps=8, max_tool_calls=1, timeout_seconds=30))
    with pytest.raises(AgentBudgetError, match="tool-call"):
        await agent.run("switch")


async def test_text_only_turn_nudges_then_recovers() -> None:
    llm = ScriptedLLM(
        [
            _text("I think you want a Switch."),
            _tools(ToolCall("s", SEARCH_TOOL, {"keyword": "switch"})),
            _tools(ToolCall("f", SUBMIT_TOOL, _submit(("m1", "m2", "m3")))),
        ]
    )
    result = await _agent(llm).run("switch")
    assert result.steps == 3
    nudge = llm.requests[1].messages[-1]
    assert nudge.role is Role.USER
    assert SEARCH_TOOL in nudge.content


async def test_empty_request_is_rejected() -> None:
    llm = ScriptedLLM([])
    with pytest.raises(AgentInputError, match="empty"):
        await _agent(llm).run("   ")
    assert llm.requests == []


async def test_build_agent_applies_settings() -> None:
    llm = ScriptedLLM(
        [
            _tools(ToolCall("s1", SEARCH_TOOL, {"keyword": "one"})),
            _tools(ToolCall("s2", SEARCH_TOOL, {"keyword": "two"})),
        ]
    )
    registry = ToolRegistry.for_source(FakeSource(result=SearchResult(items=_three())))
    agent = build_agent(
        llm,
        registry,
        settings=Settings(agent_max_steps=1, agent_timeout=30, max_user_chars=20),
    )
    assert agent.budget.max_steps == 1
    with pytest.raises(AgentBudgetError, match="step"):
        await agent.run("switch")
    with pytest.raises(AgentInputError, match="exceeds"):
        await ShopperAgent(llm, registry, input_guard=InputGuard(max_chars=4)).run("switch")


def test_load_settings_reads_agent_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_MAX_STEPS", "4")
    monkeypatch.setenv("AGENT_MAX_TOOL_CALLS", "6")
    monkeypatch.setenv("AGENT_TIMEOUT", "15")
    monkeypatch.setenv("AGENT_LLM_ATTEMPTS", "2")
    monkeypatch.setenv("AGENT_MAX_USER_CHARS", "100")
    settings = load_settings()
    assert settings.agent_max_steps == 4
    assert settings.agent_max_tool_calls == 6
    assert settings.agent_timeout == 15.0
    assert settings.agent_llm_attempts == 2
    assert settings.max_user_chars == 100
    monkeypatch.setenv("AGENT_MAX_STEPS", "0")
    with pytest.raises(ValueError, match="AGENT_MAX_STEPS"):
        load_settings()


async def test_ask_user_stops_the_loop() -> None:
    llm = ScriptedLLM(
        [
            _tools(ToolCall("q", ASK_TOOL, {"question": "What kind of gift?"})),
            _tools(ToolCall("s", SEARCH_TOOL, {"keyword": "switch"})),
        ]
    )
    result = await ShopperAgent(llm, ToolRegistry.for_source(FakeSource(), allow_clarify=True)).run(
        "a nice gift"
    )
    assert result.question == "What kind of gift?"
    assert result.recommendations == ()
    assert result.listings == ()
    assert len(llm.requests) == 1
    assert result.trace[0].tool_names == (ASK_TOOL,)


async def test_non_interactive_registry_omits_ask_user() -> None:
    registry = ToolRegistry.for_source(FakeSource())
    assert ASK_TOOL not in registry.names()
    llm = ScriptedLLM(
        [
            _tools(ToolCall("q", ASK_TOOL, {"question": "What kind of gift?"})),
            _tools(ToolCall("s", SEARCH_TOOL, {"keyword": "switch"})),
            _tools(ToolCall("f", SUBMIT_TOOL, _submit(("m1", "m2", "m3")))),
        ]
    )
    result = await _agent(llm).run("a nice gift")
    assert result.question is None
    assert [rec.item_id for rec in result.recommendations] == ["m1", "m2", "m3"]
    first = llm.requests[1].messages[-1].tool_result
    assert first is not None
    assert first.is_error is True
    assert "unknown tool" in first.content


async def test_fair_value_score_appears_on_grounded_listings() -> None:
    sold = SearchResult(
        items=[
            Listing(
                item_id=f"s{i}",
                name="Switch OLED",
                price=price,
                thumbnail_url="http://img",
            )
            for i, price in enumerate([18000, 20000, 22000, 24000, 26000, 28000, 30000], start=1)
        ]
    )
    detailed = [
        Listing(
            item_id=item.item_id,
            name=item.name,
            price=item.price,
            thumbnail_url=item.thumbnail_url,
            condition_id=3,
            seller=SellerInfo(seller_id="1", name="sample-seller", num_ratings=40, star_rating=5.0),
        )
        for item in _three()
    ]
    source = FakeSource(result=SearchResult(items=_three()), sold=sold, items=detailed)
    llm = ScriptedLLM(
        [
            _tools(ToolCall("s", SEARCH_TOOL, {"keyword": "switch"})),
            _tools(ToolCall("m", MARKET_TOOL, {"keyword": "switch"})),
            _tools(ToolCall("f", SUBMIT_TOOL, _submit(("m1", "m2", "m3")))),
        ]
    )
    result = await ShopperAgent(
        llm, ToolRegistry.for_source(source), enricher=build_enricher(source)
    ).run("switch under 30000")
    score = result.listings[0]["fair_value"]
    assert score["confidence"] == "ok"
    assert score["sample_size"] == 7
    assert "typical sold price" in score["label"]
    assert result.listings[0]["seller"]["name"] == "sample-seller"
    assert MARKET_TOOL in result.trace[1].tool_names


async def test_ask_user_is_skipped_once_listings_exist() -> None:
    llm = ScriptedLLM(
        [
            _tools(ToolCall("s", SEARCH_TOOL, {"keyword": "switch"})),
            _tools(ToolCall("q", ASK_TOOL, {"question": "Which color?"})),
            _tools(ToolCall("f", SUBMIT_TOOL, _submit(("m1", "m2", "m3")))),
        ]
    )
    source = FakeSource(result=SearchResult(items=_three()))
    result = await ShopperAgent(llm, ToolRegistry.for_source(source, allow_clarify=True)).run(
        "switch"
    )
    assert result.question is None
    assert [rec.item_id for rec in result.recommendations] == ["m1", "m2", "m3"]
    skipped = [
        message.tool_result
        for message in llm.requests[2].messages
        if message.tool_result is not None
    ]
    assert skipped
    assert skipped[-1].is_error is True
    assert "skipped" in skipped[-1].content


async def test_search_and_details_then_submit() -> None:
    source = FakeSource(result=SearchResult(items=_three()), items=_three())
    llm = ScriptedLLM(
        [
            _tools(ToolCall("s", SEARCH_TOOL, {"keyword": "switch"})),
            _tools(ToolCall("d", DETAIL_TOOL, {"item_ids": ["m1", "m2"]})),
            _tools(ToolCall("f", SUBMIT_TOOL, _submit(("m1", "m2", "m3")))),
        ]
    )
    result = await ShopperAgent(llm, ToolRegistry.for_source(source)).run("switch")
    assert source.item_calls == 1
    assert result.tool_calls == 2
