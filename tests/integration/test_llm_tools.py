"""Integration: an Anthropic tool call runs through the registry and fixtures."""

import json
from types import SimpleNamespace

import pytest
from mercari_shopper.llm.anthropic_client import AnthropicAdapter
from mercari_shopper.llm.base import Message, Role
from mercari_shopper.sources.fixture_source import FixtureSource
from mercari_shopper.tools.registry import ToolRegistry

pytestmark = pytest.mark.integration


class _Messages:
    def __init__(self, response: object) -> None:
        self.response = response
        self.kwargs: dict[str, object] | None = None

    async def create(self, **kwargs: object) -> object:
        self.kwargs = kwargs
        return self.response


class _Client:
    def __init__(self, messages: _Messages) -> None:
        self.messages = messages


async def test_scripted_tool_call_searches_fixtures_then_loads_details() -> None:
    registry = ToolRegistry.for_source(FixtureSource.from_package())
    messages = _Messages(
        SimpleNamespace(
            stop_reason="tool_use",
            content=[
                SimpleNamespace(
                    type="tool_use",
                    id="toolu_search",
                    name="search_mercari",
                    input={"keyword": "switch", "price_max": 30000, "limit": 5},
                )
            ],
        )
    )
    llm = AnthropicAdapter(model="claude-test", client=_Client(messages))
    response = await llm.complete(
        [Message(role=Role.USER, content="a used Switch under 30000 yen")],
        tools=registry.specs(),
    )

    assert messages.kwargs is not None
    sent_tools = messages.kwargs["tools"]
    assert isinstance(sent_tools, list)
    assert {tool["name"] for tool in sent_tools} == {
        "search_mercari",
        "get_item_details",
        "submit_recommendations",
    }
    assert response.tool_calls[0].argument_error is None

    search = await registry.execute(response.tool_calls[0])
    assert search.is_error is False
    found = json.loads(search.content)
    assert found["query"]["price_max"] == 30000
    assert found["items"]
    assert all(item["price"] <= 30000 for item in found["items"])

    item_id = found["items"][0]["item_id"]
    details = await registry.execute(
        type(response.tool_calls[0])(
            id="toolu_details",
            name="get_item_details",
            arguments={"item_ids": [item_id]},
        )
    )
    assert details.is_error is False
    loaded = json.loads(details.content)
    assert loaded["items"][0]["item_id"] == item_id
    assert loaded["missing_ids"] == []
    assert "seller" in loaded["items"][0] or "description" in loaded["items"][0]
