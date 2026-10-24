from types import SimpleNamespace
from typing import Any

import pytest
from mercari_shopper.llm.anthropic_client import AnthropicAdapter
from mercari_shopper.llm.base import (
    ForcedTool,
    Message,
    Role,
    ToolCall,
    ToolChoiceMode,
    ToolResult,
    ToolSpec,
)
from mercari_shopper.llm.errors import LLMError

SPEC = ToolSpec(
    name="search_mercari",
    description="Search Mercari",
    parameters={"type": "object", "properties": {"keyword": {"type": "string"}}},
)


class FakeMessages:
    def __init__(self, response: object | None = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.kwargs: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> object:
        self.kwargs.append(kwargs)
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


class FakeAnthropic:
    def __init__(self, messages: FakeMessages) -> None:
        self.messages = messages
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def adapter(messages: FakeMessages) -> AnthropicAdapter:
    return AnthropicAdapter(model="claude-test", client=FakeAnthropic(messages))


async def test_request_maps_tools_system_and_coalesced_results() -> None:
    messages = FakeMessages(response=SimpleNamespace(content=[], stop_reason="end_turn"))
    client = adapter(messages)
    transcript = [
        Message(role=Role.SYSTEM, content="system one"),
        Message(role=Role.SYSTEM, content="system two"),
        Message(role=Role.USER, content="find a switch"),
        Message(
            role=Role.ASSISTANT,
            content="Searching.",
            tool_calls=(
                ToolCall(id="toolu_1", name="search_mercari", arguments={"keyword": "有機EL"}),
                ToolCall(id="toolu_2", name="search_mercari", arguments={"keyword": "switch"}),
            ),
        ),
        Message(role=Role.TOOL, tool_result=ToolResult("toolu_1", "[]")),
        Message(role=Role.TOOL, tool_result=ToolResult("toolu_2", "failed", is_error=True)),
    ]

    await client.complete(transcript, tools=[SPEC], tool_choice=ForcedTool("search_mercari"))

    payload = messages.kwargs[0]
    assert payload["model"] == "claude-test"
    assert payload["max_tokens"] == 4096
    assert payload["system"] == "system one\n\nsystem two"
    assert payload["tools"] == [
        {
            "name": "search_mercari",
            "description": "Search Mercari",
            "input_schema": SPEC.parameters,
        }
    ]
    assert payload["tool_choice"] == {"type": "tool", "name": "search_mercari"}
    assert [item["role"] for item in payload["messages"]] == ["user", "assistant", "user"]
    tool_uses = payload["messages"][1]["content"]
    assert tool_uses[0] == {"type": "text", "text": "Searching."}
    assert tool_uses[1]["input"] == {"keyword": "有機EL"}
    results = payload["messages"][2]["content"]
    assert results[0] == {"type": "tool_result", "tool_use_id": "toolu_1", "content": "[]"}
    assert results[1]["is_error"] is True
    assert "is_error" not in results[0]


async def test_response_maps_text_and_tool_use() -> None:
    raw = SimpleNamespace(
        stop_reason="tool_use",
        content=[
            SimpleNamespace(type="thinking", thinking="hidden"),
            SimpleNamespace(type="text", text="Looking."),
            SimpleNamespace(
                type="tool_use",
                id="toolu_9",
                name="search_mercari",
                input={"keyword": "switch", "limit": 5},
            ),
        ],
    )
    response = await adapter(FakeMessages(response=raw)).complete(
        [Message(role=Role.USER, content="go")],
        tools=[SPEC],
        tool_choice=ToolChoiceMode.ANY,
    )
    assert response.stop_reason == "tool_use"
    assert response.text == "Looking."
    assert response.tool_calls == (
        ToolCall(
            id="toolu_9",
            name="search_mercari",
            arguments={"keyword": "switch", "limit": 5},
        ),
    )
    assert adapter(FakeMessages(response=raw)).build_payload(
        [Message(role=Role.USER, content="go")],
        tools=[SPEC],
        tool_choice=ToolChoiceMode.ANY,
    )["tool_choice"] == {"type": "any"}


async def test_invalid_tool_input_becomes_empty_arguments() -> None:
    raw = SimpleNamespace(
        stop_reason="tool_use",
        content=[SimpleNamespace(type="tool_use", id="t", name="search_mercari", input="nope")],
    )
    response = await adapter(FakeMessages(response=raw)).complete(
        [Message(role=Role.USER, content="go")]
    )
    assert response.tool_calls[0].arguments == {}
    assert response.tool_calls[0].argument_error is not None
    assert "JSON" in response.tool_calls[0].argument_error


async def test_omits_tools_when_none_are_offered() -> None:
    messages = FakeMessages(response=SimpleNamespace(content=[], stop_reason="end_turn"))
    await adapter(messages).complete([Message(role=Role.USER, content="hi")])
    assert "tools" not in messages.kwargs[0]
    assert "tool_choice" not in messages.kwargs[0]
    assert "system" not in messages.kwargs[0]


async def test_bad_transcript_does_not_call_the_sdk() -> None:
    messages = FakeMessages()
    with pytest.raises(ValueError, match="tool message"):
        await adapter(messages).complete([Message(role=Role.TOOL, content="x")])
    assert messages.kwargs == []


async def test_forced_tool_must_be_advertised() -> None:
    with pytest.raises(ValueError, match="not in the tool list"):
        await adapter(FakeMessages()).complete(
            [Message(role=Role.USER, content="go")],
            tools=[SPEC],
            tool_choice=ForcedTool("missing"),
        )


async def test_provider_errors_are_classified() -> None:
    with pytest.raises(LLMError) as limited:
        await adapter(FakeMessages(error=_StatusError(429, "slow"))).complete(
            [Message(role=Role.USER, content="go")]
        )
    assert limited.value.retryable is True
    assert limited.value.status == 429
    assert limited.value.provider == "anthropic"

    with pytest.raises(LLMError) as rejected:
        await adapter(FakeMessages(error=_StatusError(400, "bad"))).complete(
            [Message(role=Role.USER, content="go")]
        )
    assert rejected.value.retryable is False

    with pytest.raises(LLMError) as timed_out:
        await adapter(FakeMessages(error=TimeoutError("timed out"))).complete(
            [Message(role=Role.USER, content="go")]
        )
    assert timed_out.value.retryable is True
    assert timed_out.value.status is None

    with pytest.raises(LLMError) as crashed:
        await adapter(FakeMessages(error=RuntimeError("bug"))).complete(
            [Message(role=Role.USER, content="go")]
        )
    assert crashed.value.retryable is False


async def test_consecutive_user_turns_are_merged() -> None:
    messages = FakeMessages(response=SimpleNamespace(content=[], stop_reason="end_turn"))
    await adapter(messages).complete(
        [
            Message(role=Role.USER, content="find a switch"),
            Message(
                role=Role.ASSISTANT,
                tool_calls=(
                    ToolCall(id="t", name="search_mercari", arguments={"keyword": "switch"}),
                ),
            ),
            Message(role=Role.TOOL, tool_result=ToolResult("t", "[]")),
            Message(role=Role.USER, content="none of those, try OLED"),
        ]
    )
    sent = messages.kwargs[0]["messages"]
    assert [item["role"] for item in sent] == ["user", "assistant", "user"]
    repair = sent[2]["content"]
    assert repair[0]["type"] == "tool_result"
    assert repair[-1] == {"type": "text", "text": "none of those, try OLED"}


async def test_unreadable_response_raises() -> None:
    with pytest.raises(LLMError, match="no content"):
        await adapter(FakeMessages(response=SimpleNamespace(content=None))).complete(
            [Message(role=Role.USER, content="go")]
        )


async def test_injected_client_is_not_closed() -> None:
    fake = FakeAnthropic(FakeMessages())
    client = AnthropicAdapter(model="claude-test", client=fake)
    await client.aclose()
    assert fake.closed is False


async def test_owned_client_is_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    holder: dict[str, Any] = {}

    class FakeSDK:
        def __init__(self, *, api_key: str, timeout: float) -> None:
            holder["api_key"] = api_key
            holder["timeout"] = timeout
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    monkeypatch.setattr("mercari_shopper.llm.anthropic_client.AsyncAnthropic", FakeSDK)
    client = AnthropicAdapter(model="claude-test", api_key="sk-ant", timeout=12)
    await client.aclose()
    assert holder["api_key"] == "sk-ant"
    assert holder["timeout"] == 12
    assert client._client.closed is True


class _StatusError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
