from types import SimpleNamespace

import pytest
from mercari_shopper.llm.base import Message, Role, ToolChoiceMode, ToolSpec
from mercari_shopper.llm.vertex_client import AnthropicVertexAdapter

SPEC = ToolSpec(
    name="search_mercari",
    description="Search Mercari",
    parameters={"type": "object", "properties": {"keyword": {"type": "string"}}},
)


class FakeMessages:
    def __init__(self, response: object) -> None:
        self.response = response
        self.kwargs: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> object:
        self.kwargs.append(kwargs)
        return self.response


class FakeVertex:
    def __init__(self, messages: FakeMessages) -> None:
        self.messages = messages


async def test_vertex_sends_the_anthropic_message_shape() -> None:
    messages = FakeMessages(response=SimpleNamespace(content=[], stop_reason="end_turn"))
    client = AnthropicVertexAdapter(
        model="claude-sonnet-4-5@20250929",
        client=FakeVertex(messages),
    )
    await client.complete(
        [
            Message(role=Role.SYSTEM, content="be brief"),
            Message(role=Role.USER, content="find a switch"),
        ],
        tools=[SPEC],
        tool_choice=ToolChoiceMode.ANY,
    )
    payload = messages.kwargs[0]
    assert payload["model"] == "claude-sonnet-4-5@20250929"
    assert payload["system"] == [
        {"type": "text", "text": "be brief", "cache_control": {"type": "ephemeral"}}
    ]
    assert payload["tool_choice"] == {"type": "any"}
    assert payload["tools"] == [
        {
            "name": "search_mercari",
            "description": "Search Mercari",
            "input_schema": SPEC.parameters,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    assert payload["messages"] == [{"role": "user", "content": "find a switch"}]


def test_missing_google_auth_fails_before_the_request(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing() -> None:
        raise ValueError("Vertex AI needs google-auth")

    monkeypatch.setattr("mercari_shopper.llm.vertex_client._require_google_auth", missing)
    with pytest.raises(ValueError, match="google-auth"):
        AnthropicVertexAdapter(model="claude-haiku-5-5", project_id="p", region="global")


async def test_owned_vertex_client_is_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSDK:
        def __init__(self, *, project_id: str, region: str, timeout: float) -> None:
            self.project_id = project_id
            self.region = region
            self.timeout = timeout
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    monkeypatch.setattr("mercari_shopper.llm.vertex_client.AsyncAnthropicVertex", FakeSDK)
    client = AnthropicVertexAdapter(
        model="claude-sonnet-4-5@20250929", project_id="p", region="us-east5"
    )
    assert client.provider == "vertex"
    await client.aclose()
    assert client._client.closed is True
    assert client._client.project_id == "p"
