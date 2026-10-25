import json
from types import SimpleNamespace
from typing import Any

import pytest
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
from mercari_shopper.llm.openai_client import OpenAIAdapter

SPEC = ToolSpec(
    name="search_mercari",
    description="Search Mercari",
    parameters={"type": "object", "properties": {"keyword": {"type": "string"}}},
)


class FakeCompletions:
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


class FakeOpenAI:
    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def adapter(completions: FakeCompletions) -> OpenAIAdapter:
    return OpenAIAdapter(model="gpt-test", client=FakeOpenAI(completions))


async def test_request_maps_messages_and_function_tools() -> None:
    completions = FakeCompletions(
        response=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok", tool_calls=None), finish_reason="stop"
                )
            ]
        )
    )
    transcript = [
        Message(role=Role.SYSTEM, content="be brief"),
        Message(role=Role.USER, content="find a switch"),
        Message(
            role=Role.ASSISTANT,
            tool_calls=(
                ToolCall(id="call_1", name="search_mercari", arguments={"keyword": "有機EL"}),
            ),
        ),
        Message(role=Role.TOOL, tool_result=ToolResult("call_1", '{"returned":0}')),
    ]
    await adapter(completions).complete(
        transcript,
        tools=[SPEC],
        tool_choice=ToolChoiceMode.ANY,
    )
    payload = completions.kwargs[0]
    assert payload["model"] == "gpt-test"
    assert payload["max_completion_tokens"] == 4096
    assert payload["tool_choice"] == "required"
    assert payload["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "search_mercari",
                "description": "Search Mercari",
                "parameters": SPEC.parameters,
            },
        }
    ]
    system, user, assistant, tool = payload["messages"]
    assert system == {"role": "system", "content": "be brief"}
    assert user == {"role": "user", "content": "find a switch"}
    assert assistant["content"] is None
    arguments = assistant["tool_calls"][0]["function"]["arguments"]
    assert json.loads(arguments) == {"keyword": "有機EL"}
    assert "\\u" not in arguments
    assert tool == {"role": "tool", "tool_call_id": "call_1", "content": '{"returned":0}'}


async def test_forced_tool_choice_names_the_function() -> None:
    completions = FakeCompletions(
        response=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="", tool_calls=None), finish_reason="stop"
                )
            ]
        )
    )
    await adapter(completions).complete(
        [Message(role=Role.USER, content="go")],
        tools=[SPEC],
        tool_choice=ForcedTool("search_mercari"),
    )
    assert completions.kwargs[0]["tool_choice"] == {
        "type": "function",
        "function": {"name": "search_mercari"},
    }


async def test_response_maps_tool_calls_and_invalid_json() -> None:
    raw = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call_9",
                            function=SimpleNamespace(
                                name="search_mercari",
                                arguments='{"keyword":"switch"}',
                            ),
                        ),
                        SimpleNamespace(
                            id="call_10",
                            function=SimpleNamespace(name="search_mercari", arguments="{"),
                        ),
                    ],
                ),
            )
        ]
    )
    response = await adapter(FakeCompletions(response=raw)).complete(
        [Message(role=Role.USER, content="go")],
        tools=[SPEC],
    )
    assert response.stop_reason == "tool_calls"
    assert response.text == ""
    assert response.tool_calls[0].arguments == {"keyword": "switch"}
    assert response.tool_calls[0].argument_error is None
    assert response.tool_calls[1].arguments == {}
    assert response.tool_calls[1].argument_error is not None
    assert "JSON" in response.tool_calls[1].argument_error


async def test_empty_choices_raise() -> None:
    raw = SimpleNamespace(choices=[])
    with pytest.raises(LLMError, match="no choices"):
        await adapter(FakeCompletions(response=raw)).complete(
            [Message(role=Role.USER, content="go")]
        )


async def test_missing_function_raises() -> None:
    raw = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content=None, tool_calls=[SimpleNamespace(id="x", function=None)]
                ),
            )
        ]
    )
    with pytest.raises(LLMError, match="function"):
        await adapter(FakeCompletions(response=raw)).complete(
            [Message(role=Role.USER, content="go")]
        )


async def test_status_error_is_not_retried_on_400() -> None:
    error = Exception("bad request")
    error.status_code = 400  # type: ignore[attr-defined]
    with pytest.raises(LLMError) as caught:
        await adapter(FakeCompletions(error=error)).complete(
            [Message(role=Role.USER, content="go")]
        )
    assert caught.value.retryable is False
    assert caught.value.provider == "openai"


async def test_none_choice_is_forwarded() -> None:
    completions = FakeCompletions(
        response=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="hi", tool_calls=None), finish_reason="stop"
                )
            ]
        )
    )
    await adapter(completions).complete(
        [Message(role=Role.USER, content="go")],
        tools=[SPEC],
        tool_choice=ToolChoiceMode.NONE,
    )
    assert completions.kwargs[0]["tool_choice"] == "none"


async def test_effort_is_skipped_on_non_reasoning_models() -> None:
    """gpt-4.1 (the default model) does not support reasoning_effort; sending
    it risks a 400 from the real API, so the adapter must omit it."""
    completions = FakeCompletions(
        response=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok", tool_calls=None), finish_reason="stop"
                )
            ]
        )
    )
    client = OpenAIAdapter(model="gpt-4.1", client=FakeOpenAI(completions), effort="low")
    await client.complete([Message(role=Role.USER, content="go")])
    assert "reasoning_effort" not in completions.kwargs[0]


async def test_warm_is_a_noop_for_api_key_auth() -> None:
    completions = FakeCompletions(
        response=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok", tool_calls=None), finish_reason="stop"
                )
            ]
        )
    )
    fake = FakeOpenAI(completions)
    client = OpenAIAdapter(model="gpt-4.1", client=fake)
    await client.warm()
    await client.warm()
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

    monkeypatch.setattr("mercari_shopper.llm.openai_client.AsyncOpenAI", FakeSDK)
    client = OpenAIAdapter(model="gpt-4.1", api_key="sk-live", timeout=9)
    await client.aclose()
    assert holder == {"api_key": "sk-live", "timeout": 9}
    assert client._client.closed is True


async def test_effort_is_sent_on_reasoning_models() -> None:
    completions = FakeCompletions(
        response=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok", tool_calls=None), finish_reason="stop"
                )
            ]
        )
    )
    client = OpenAIAdapter(model="gpt-5-mini", client=FakeOpenAI(completions), effort="low")
    await client.complete([Message(role=Role.USER, content="go")])
    assert completions.kwargs[0]["reasoning_effort"] == "low"
