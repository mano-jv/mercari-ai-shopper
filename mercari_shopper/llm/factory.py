"""Select an LLM adapter from LLM_PROVIDER or an explicit name."""

from __future__ import annotations

from typing import Any

from mercari_shopper.config import Settings, load_settings
from mercari_shopper.llm.anthropic_client import AnthropicAdapter
from mercari_shopper.llm.base import LLMClient
from mercari_shopper.llm.openai_client import OpenAIAdapter
from mercari_shopper.llm.vertex_client import AnthropicVertexAdapter

# Mid-tier defaults. Set LLM_MODEL when a provider renames a model.
# Vertex ids include the publisher version after @.
DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-5",
    "vertex": "claude-sonnet-4-5@20250929",
    "openai": "gpt-4.1",
}

_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def build_llm(
    provider: str | None = None,
    *,
    settings: Settings | None = None,
    api_key: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
    client: Any | None = None,
) -> LLMClient:
    """Factory. Pass `client` in tests so no SDK client is constructed."""
    resolved = settings or load_settings()
    name = (provider or resolved.llm_provider).strip().lower()
    if name not in DEFAULT_MODELS:
        raise ValueError(f"unknown LLM provider {name!r}")
    chosen_model = _choose_model(model, resolved.llm_model, name)
    chosen_tokens = resolved.llm_max_tokens if max_tokens is None else max_tokens
    chosen_timeout = resolved.llm_timeout if timeout is None else timeout
    if name == "vertex":
        _require_vertex(resolved, client)
        return AnthropicVertexAdapter(
            model=chosen_model,
            project_id=resolved.vertex_project_id,
            region=resolved.vertex_region,
            max_tokens=chosen_tokens,
            timeout=chosen_timeout,
            client=client,
        )
    key = _choose_key(name, resolved, api_key, client)
    if name == "anthropic":
        return AnthropicAdapter(
            model=chosen_model,
            api_key=key,
            max_tokens=chosen_tokens,
            timeout=chosen_timeout,
            client=client,
        )
    return OpenAIAdapter(
        model=chosen_model,
        api_key=key,
        max_tokens=chosen_tokens,
        timeout=chosen_timeout,
        client=client,
    )


def _choose_model(explicit: str | None, configured: str, provider: str) -> str:
    chosen = configured if explicit is None else explicit
    stripped = chosen.strip()
    if stripped:
        return stripped
    return DEFAULT_MODELS[provider]


def _choose_key(provider: str, settings: Settings, api_key: str | None, client: Any | None) -> str:
    if provider not in _KEY_ENV:
        raise ValueError(f"{provider} does not use an API key")
    if api_key is not None:
        key = api_key.strip()
    elif provider == "anthropic":
        key = settings.anthropic_api_key
    else:
        key = settings.openai_api_key
    if client is None and not key:
        raise ValueError(f"{_KEY_ENV[provider]} is not set")
    return key


def _require_vertex(settings: Settings, client: Any | None) -> None:
    if client is not None:
        return
    missing: list[str] = []
    if not settings.vertex_project_id:
        missing.append("ANTHROPIC_VERTEX_PROJECT_ID or GOOGLE_CLOUD_PROJECT")
    if not settings.vertex_region:
        missing.append("CLOUD_ML_REGION")
    if missing:
        raise ValueError("Vertex AI is missing " + " and ".join(missing))
