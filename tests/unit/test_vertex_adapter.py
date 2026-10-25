from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from mercari_shopper.llm.base import Message, Role, ToolChoiceMode, ToolSpec
from mercari_shopper.llm.vertex_client import (
    AnthropicVertexAdapter,
    _adc_file_path,
    _shared_credentials,
)

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


class FakeCredentials:
    def __init__(self, *, token: str | None = None, expired: bool = True) -> None:
        self.token = token
        self.expired = expired
        self.refreshed = 0

    def refresh(self, request: object) -> None:
        del request
        self.refreshed += 1
        self.token = "tok"
        self.expired = False


@pytest.fixture(autouse=True)
def _clear_credential_cache() -> Iterator[None]:
    _shared_credentials.cache_clear()
    yield
    _shared_credentials.cache_clear()


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
    assert payload["messages"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "find a switch",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    ]


def test_missing_google_auth_fails_before_the_request(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing() -> None:
        raise ValueError("Vertex AI needs google-auth")

    monkeypatch.setattr("mercari_shopper.llm.vertex_client._require_google_auth", missing)
    with pytest.raises(ValueError, match="google-auth"):
        AnthropicVertexAdapter(model="claude-haiku-5-5", project_id="p", region="global")


def test_adc_file_path_prefers_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", r"C:\creds\adc.json")
    monkeypatch.setattr(
        "mercari_shopper.llm.vertex_client._well_known_adc_path",
        lambda: r"C:\Users\me\AppData\Roaming\gcloud\application_default_credentials.json",
    )
    assert _adc_file_path() == r"C:\creds\adc.json"


def test_adc_file_path_uses_well_known_file_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.setattr(
        "mercari_shopper.llm.vertex_client._well_known_adc_path",
        lambda: "/home/me/.config/gcloud/application_default_credentials.json",
    )
    assert _adc_file_path() == "/home/me/.config/gcloud/application_default_credentials.json"


def test_shared_credentials_loads_the_adc_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "adc.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(path))
    sentinel = object()
    loads: list[tuple[str, list[str] | None]] = []

    def fake_load(filename: str, scopes: list[str] | None = None) -> tuple[object, None]:
        loads.append((filename, scopes))
        return sentinel, None

    monkeypatch.setattr(
        "google.auth.load_credentials_from_file",
        fake_load,
    )
    assert _shared_credentials() is sentinel
    assert _shared_credentials() is sentinel
    assert loads == [(str(path), ["https://www.googleapis.com/auth/cloud-platform"])]


def test_shared_credentials_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.setattr("mercari_shopper.llm.vertex_client._well_known_adc_path", lambda: None)
    sentinel = object()
    called: list[list[str] | None] = []

    def fake_default(scopes: list[str] | None = None) -> tuple[object, str]:
        called.append(scopes)
        return sentinel, "proj"

    monkeypatch.setattr("google.auth.default", fake_default)
    assert _shared_credentials() is sentinel
    assert called == [["https://www.googleapis.com/auth/cloud-platform"]]


async def test_owned_vertex_client_is_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    creds = FakeCredentials()

    class FakeSDK:
        def __init__(
            self,
            *,
            project_id: str,
            region: str,
            timeout: float,
            credentials: object | None = None,
        ) -> None:
            self.project_id = project_id
            self.region = region
            self.timeout = timeout
            self.credentials = credentials
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    monkeypatch.setattr("mercari_shopper.llm.vertex_client._shared_credentials", lambda: creds)
    monkeypatch.setattr("mercari_shopper.llm.vertex_client.AsyncAnthropicVertex", FakeSDK)
    client = AnthropicVertexAdapter(
        model="claude-sonnet-4-5@20250929", project_id="p", region="us-east5"
    )
    assert client.provider == "vertex"
    assert client._client.credentials is creds
    await client.aclose()
    assert client._client.closed is True
    assert client._client.project_id == "p"


async def test_warm_refreshes_expired_token_once(monkeypatch: pytest.MonkeyPatch) -> None:
    creds = FakeCredentials(token=None, expired=True)

    class FakeSDK:
        def __init__(
            self,
            *,
            project_id: str,
            region: str,
            timeout: float,
            credentials: object | None = None,
        ) -> None:
            self.credentials = credentials

    monkeypatch.setattr("mercari_shopper.llm.vertex_client._shared_credentials", lambda: creds)
    monkeypatch.setattr("mercari_shopper.llm.vertex_client.AsyncAnthropicVertex", FakeSDK)
    client = AnthropicVertexAdapter(model="claude-haiku-5-5", project_id="p", region="global")
    await client.warm()
    await client.warm()
    assert creds.refreshed == 1
    assert creds.token == "tok"


async def test_warm_skips_injected_client() -> None:
    messages = FakeMessages(response=SimpleNamespace(content=[], stop_reason="end_turn"))
    client = AnthropicVertexAdapter(model="claude-haiku-5-5", client=FakeVertex(messages))
    await client.warm()
    assert not hasattr(client._client, "credentials")


async def test_vertex_forwards_effort_to_the_inherited_payload() -> None:
    messages = FakeMessages(response=SimpleNamespace(content=[], stop_reason="end_turn"))
    client = AnthropicVertexAdapter(
        model="claude-haiku-5-5",
        client=FakeVertex(messages),
        effort="low",
    )
    await client.complete([Message(role=Role.USER, content="go")])
    assert messages.kwargs[0]["output_config"] == {"effort": "low"}
