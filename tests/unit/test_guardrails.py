import pytest
from mercari_shopper.agent.budget import AgentBudget
from mercari_shopper.agent.errors import AgentBudgetError, AgentInputError
from mercari_shopper.agent.guardrails import (
    UNTRUSTED_END,
    UNTRUSTED_START,
    GroundingValidator,
    InputGuard,
    wrap_untrusted,
)
from mercari_shopper.agent.prompts import PromptBuilder
from mercari_shopper.config import Settings
from mercari_shopper.tools.registry import ToolExecution
from mercari_shopper.tools.schemas import SEARCH_TOOL, SUBMIT_TOOL


def test_input_guard_rejects_empty_and_oversized() -> None:
    guard = InputGuard(max_chars=8)
    assert guard.sanitize("  switch  ") == "switch"
    with pytest.raises(AgentInputError, match="empty"):
        guard.sanitize("   ")
    with pytest.raises(AgentInputError, match="exceeds"):
        guard.sanitize("123456789")


def test_wrap_untrusted_uses_delimiters_the_prompt_names() -> None:
    wrapped = wrap_untrusted('{"item_id":"m1"}')
    assert wrapped.startswith(UNTRUSTED_START)
    assert wrapped.endswith(UNTRUSTED_END)
    assert '{"item_id":"m1"}' in wrapped
    prompt = PromptBuilder().system()
    assert UNTRUSTED_START in prompt
    assert UNTRUSTED_END in prompt
    assert "untrusted" in prompt.lower()
    assert SEARCH_TOOL in prompt
    assert SUBMIT_TOOL in prompt
    assert "1:" in prompt
    assert "Japanese" in prompt
    assert "two search turns" in PromptBuilder().system()
    assert SUBMIT_TOOL in PromptBuilder().nudge_submit()


def test_grounding_validator_accepts_ids_from_search_only() -> None:
    validator = GroundingValidator()
    validator.observe(
        ToolExecution(
            tool_call_id="s",
            name=SEARCH_TOOL,
            content="{}",
            is_error=False,
            data={"items": [{"item_id": "m1", "name": "A", "price": 1}]},
        )
    )
    validator.observe(
        ToolExecution(
            tool_call_id="bad",
            name=SEARCH_TOOL,
            content="{}",
            is_error=True,
            data={"items": [{"item_id": "ghost", "name": "no", "price": 1}]},
        )
    )
    ok = validator.check_submit(
        {
            "summary": "one",
            "recommendations": [
                {"item_id": "m1", "rank": 1, "reason": "a"},
                {"item_id": "m1", "rank": 2, "reason": "b"},
                {"item_id": "m1", "rank": 3, "reason": "c"},
            ],
        }
    )
    assert ok.ok is True
    assert validator.listing("m1") == {"item_id": "m1", "name": "A", "price": 1}

    bad = validator.check_submit(
        {
            "summary": "hallucinated",
            "recommendations": [
                {"item_id": "m1", "rank": 1, "reason": "a"},
                {"item_id": "nope", "rank": 2, "reason": "b"},
                {"item_id": "m1", "rank": 3, "reason": "c"},
            ],
        }
    )
    assert bad.ok is False
    assert bad.unknown_ids == ("nope",)
    assert "m1" in bad.known_ids


def test_budget_from_settings_and_limits() -> None:
    budget = AgentBudget.from_settings(
        Settings(agent_max_steps=2, agent_max_tool_calls=3, agent_timeout=10)
    )
    budget.check_turn(steps=1, elapsed=1.0)
    with pytest.raises(AgentBudgetError, match="step"):
        budget.check_turn(steps=2, elapsed=1.0)
    with pytest.raises(AgentBudgetError, match="time"):
        budget.check_turn(steps=0, elapsed=10.0)
    budget.check_tools(2, 1)
    with pytest.raises(AgentBudgetError, match="tool-call"):
        budget.check_tools(2, 2)
    budget.check_tools(3, 0)
    with pytest.raises(ValueError, match="max_steps"):
        AgentBudget(max_steps=0)
