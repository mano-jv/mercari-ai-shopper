from typing import Any

import pytest
from mercari_shopper.config import DEFAULT_LLM_MAX_TOKENS, Settings, load_settings
from mercari_shopper.llm.anthropic_client import AnthropicAdapter
from mercari_shopper.llm.factory import DEFAULT_MODELS, build_llm
from mercari_shopper.llm.openai_client import OpenAIAdapter
from mercari_shopper.llm.vertex_client import AnthropicVertexAdapter


def test_load_settings_reads_provider_and_rejects_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    settings = load_settings()
    assert settings.llm_provider == "openai"
    assert settings.llm_model == "gpt-test"
    assert settings.openai_api_key == "sk-test"
    assert "sk-test" not in repr(settings)

    monkeypatch.setenv("LLM_PROVIDER", "ftp")
    with pytest.raises(ValueError, match="ftp"):
        load_settings()


def test_load_settings_rejects_bad_token_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MAX_TOKENS", "0")
    with pytest.raises(ValueError, match="LLM_MAX_TOKENS"):
        load_settings()
    monkeypatch.setenv("LLM_MAX_TOKENS", "nope")
    with pytest.raises(ValueError, match="not an integer"):
        load_settings()


def test_effort_defaults_to_low_and_rejects_unknown_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLM_EFFORT", raising=False)
    assert Settings().llm_effort == "low"
    assert load_settings().llm_effort == "low"

    monkeypatch.setenv("LLM_EFFORT", "medium")
    assert load_settings().llm_effort == "medium"

    monkeypatch.setenv("LLM_EFFORT", "turbo")
    with pytest.raises(ValueError, match="LLM_EFFORT"):
        load_settings()


def test_build_llm_threads_effort_into_every_provider() -> None:
    anthropic = build_llm(
        "anthropic", settings=Settings(llm_effort="low"), client=object()
    )
    openai = build_llm("openai", settings=Settings(llm_effort="low"), client=object())
    vertex = build_llm("vertex", settings=Settings(llm_effort="low"), client=object())
    assert anthropic._effort == "low"
    assert openai._effort == "low"
    assert vertex._effort == "low"


def test_build_llm_selects_adapter_and_default_model() -> None:
    anthropic = build_llm("anthropic", settings=Settings(), client=object())
    openai = build_llm("openai", settings=Settings(llm_model="gpt-custom"), client=object())
    assert isinstance(anthropic, AnthropicAdapter)
    assert isinstance(openai, OpenAIAdapter)
    assert anthropic.model == DEFAULT_MODELS["anthropic"]
    assert openai.model == "gpt-custom"


def test_claude_latency_default_is_haiku() -> None:
    assert DEFAULT_MODELS["anthropic"] == "claude-haiku-5-5"
    assert DEFAULT_MODELS["vertex"] == "claude-haiku-5-5"
    vertex = build_llm("vertex", settings=Settings(), client=object())
    assert vertex.model == "claude-haiku-5-5"
    opus = build_llm("vertex", settings=Settings(), client=object(), model="claude-opus-4-6")
    assert opus.model == "claude-opus-4-6"


def test_default_token_budget_fits_submit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_MAX_TOKENS", raising=False)
    assert DEFAULT_LLM_MAX_TOKENS == 4096
    assert load_settings().llm_max_tokens == 4096
    anthropic = build_llm("anthropic", settings=Settings(), client=object())
    assert anthropic._max_tokens == 4096


def test_missing_api_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        build_llm("anthropic", settings=Settings())
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_llm("openai", settings=Settings())


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown LLM provider"):
        build_llm("cohere", settings=Settings())


def test_factory_passes_the_key_to_the_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    created: dict[str, Any] = {}

    class FakeSDK:
        def __init__(self, *, api_key: str, timeout: float) -> None:
            created["api_key"] = api_key
            created["timeout"] = timeout

    monkeypatch.setattr("mercari_shopper.llm.openai_client.AsyncOpenAI", FakeSDK)
    client = build_llm(
        "openai",
        settings=Settings(openai_api_key="sk-live", llm_timeout=15, llm_max_tokens=256),
    )
    assert isinstance(client, OpenAIAdapter)
    assert created == {"api_key": "sk-live", "timeout": 15}
    assert client.model == DEFAULT_MODELS["openai"]


def test_vertex_uses_project_and_region_instead_of_an_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "vertex")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "from-adc")
    monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "shopper-test")
    monkeypatch.setenv("GOOGLE_CLOUD_REGION", "europe-west1")
    monkeypatch.delenv("CLOUD_ML_REGION", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = load_settings()
    assert settings.llm_provider == "vertex"
    assert settings.vertex_project_id == "shopper-test"
    assert settings.vertex_region == "europe-west1"

    created: dict[str, Any] = {}

    class FakeVertex:
        def __init__(self, *, project_id: str, region: str, timeout: float) -> None:
            created["project_id"] = project_id
            created["region"] = region
            created["timeout"] = timeout

    monkeypatch.setattr("mercari_shopper.llm.vertex_client.AsyncAnthropicVertex", FakeVertex)
    client = build_llm(settings=settings)
    assert isinstance(client, AnthropicVertexAdapter)
    assert client.model == DEFAULT_MODELS["vertex"]
    assert created == {"project_id": "shopper-test", "region": "europe-west1", "timeout": 60.0}


def test_vertex_requires_project_and_region() -> None:
    with pytest.raises(ValueError, match="ANTHROPIC_VERTEX_PROJECT_ID"):
        build_llm("vertex", settings=Settings(vertex_region="us-east5"))
    with pytest.raises(ValueError, match="CLOUD_ML_REGION"):
        build_llm("vertex", settings=Settings(vertex_project_id="shopper-test"))


def test_vertex_accepts_an_injected_client_without_credentials() -> None:
    client = build_llm("vertex", settings=Settings(), client=object())
    assert isinstance(client, AnthropicVertexAdapter)
    assert client.model == DEFAULT_MODELS["vertex"]
