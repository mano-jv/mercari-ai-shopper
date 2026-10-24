"""Runtime settings. Secrets stay in the environment, never in files."""

from __future__ import annotations

import os
from dataclasses import dataclass

_VALID_SOURCES = frozenset({"auto", "dpop", "browser", "fixture"})
_VALID_PROVIDERS = frozenset({"anthropic", "openai", "vertex"})

# Headroom above a real submit_recommendations turn (~1100 output tokens).
DEFAULT_LLM_MAX_TOKENS = 4096


@dataclass(frozen=True)
class Settings:
    mercari_source: str = "auto"
    min_interval: float = 1.0
    max_concurrent: int = 2
    cache_ttl: float = 300.0
    request_timeout: float = 20.0
    llm_provider: str = "anthropic"
    llm_model: str = ""
    llm_max_tokens: int = DEFAULT_LLM_MAX_TOKENS
    llm_timeout: float = 60.0
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    vertex_project_id: str = ""
    vertex_region: str = ""
    agent_max_steps: int = 8
    agent_max_tool_calls: int = 12
    agent_timeout: float = 90.0
    agent_llm_attempts: int = 3
    max_user_chars: int = 4000

    def __repr__(self) -> str:
        anthropic = "set" if self.anthropic_api_key else "unset"
        openai = "set" if self.openai_api_key else "unset"
        return (
            "Settings("
            f"mercari_source={self.mercari_source!r}, "
            f"min_interval={self.min_interval!r}, "
            f"max_concurrent={self.max_concurrent!r}, "
            f"cache_ttl={self.cache_ttl!r}, "
            f"request_timeout={self.request_timeout!r}, "
            f"llm_provider={self.llm_provider!r}, "
            f"llm_model={self.llm_model!r}, "
            f"llm_max_tokens={self.llm_max_tokens!r}, "
            f"llm_timeout={self.llm_timeout!r}, "
            f"anthropic_api_key={anthropic}, "
            f"openai_api_key={openai}, "
            f"vertex_project_id={self.vertex_project_id!r}, "
            f"vertex_region={self.vertex_region!r}, "
            f"agent_max_steps={self.agent_max_steps!r}, "
            f"agent_max_tool_calls={self.agent_max_tool_calls!r}, "
            f"agent_timeout={self.agent_timeout!r}, "
            f"agent_llm_attempts={self.agent_llm_attempts!r}, "
            f"max_user_chars={self.max_user_chars!r})"
        )


def load_settings() -> Settings:
    source = os.environ.get("MERCARI_SOURCE", "auto").strip().lower() or "auto"
    if source not in _VALID_SOURCES:
        raise ValueError(f"MERCARI_SOURCE={source!r} is not one of {sorted(_VALID_SOURCES)}")
    provider = os.environ.get("LLM_PROVIDER", "anthropic").strip().lower() or "anthropic"
    if provider not in _VALID_PROVIDERS:
        raise ValueError(f"LLM_PROVIDER={provider!r} is not one of {sorted(_VALID_PROVIDERS)}")
    max_tokens = _env_int("LLM_MAX_TOKENS", DEFAULT_LLM_MAX_TOKENS)
    timeout = _env_float("LLM_TIMEOUT", 60.0)
    max_steps = _env_int("AGENT_MAX_STEPS", 8)
    max_tool_calls = _env_int("AGENT_MAX_TOOL_CALLS", 12)
    agent_timeout = _env_float("AGENT_TIMEOUT", 90.0)
    llm_attempts = _env_int("AGENT_LLM_ATTEMPTS", 3)
    max_user_chars = _env_int("AGENT_MAX_USER_CHARS", 4000)
    max_concurrent = _env_int("MERCARI_MAX_CONCURRENT", 2)
    if max_tokens < 1:
        raise ValueError("LLM_MAX_TOKENS must be at least 1")
    if timeout <= 0:
        raise ValueError("LLM_TIMEOUT must be positive")
    if max_steps < 1:
        raise ValueError("AGENT_MAX_STEPS must be at least 1")
    if max_concurrent < 1:
        raise ValueError("MERCARI_MAX_CONCURRENT must be at least 1")
    if max_tool_calls < 1:
        raise ValueError("AGENT_MAX_TOOL_CALLS must be at least 1")
    if agent_timeout <= 0:
        raise ValueError("AGENT_TIMEOUT must be positive")
    if llm_attempts < 1:
        raise ValueError("AGENT_LLM_ATTEMPTS must be at least 1")
    if max_user_chars < 1:
        raise ValueError("AGENT_MAX_USER_CHARS must be at least 1")
    return Settings(
        mercari_source=source,
        min_interval=_env_float("MERCARI_MIN_INTERVAL", 1.0),
        max_concurrent=max_concurrent,
        cache_ttl=_env_float("MERCARI_CACHE_TTL", 300.0),
        request_timeout=_env_float("MERCARI_TIMEOUT", 20.0),
        llm_provider=provider,
        llm_model=os.environ.get("LLM_MODEL", "").strip(),
        llm_max_tokens=max_tokens,
        llm_timeout=timeout,
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", "").strip(),
        openai_api_key=os.environ.get("OPENAI_API_KEY", "").strip(),
        vertex_project_id=_env_first("ANTHROPIC_VERTEX_PROJECT_ID", "GOOGLE_CLOUD_PROJECT"),
        vertex_region=_env_first("CLOUD_ML_REGION", "GOOGLE_CLOUD_REGION"),
        agent_max_steps=max_steps,
        agent_max_tool_calls=max_tool_calls,
        agent_timeout=agent_timeout,
        agent_llm_attempts=llm_attempts,
        max_user_chars=max_user_chars,
    )


def _env_first(*names: str) -> str:
    for name in names:
        raw = os.environ.get(name)
        if raw is not None and raw.strip():
            return raw.strip()
    return ""


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not an integer") from exc
