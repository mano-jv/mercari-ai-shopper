"""Runtime settings. Secrets stay in the environment, never in files."""

from __future__ import annotations

import os
from dataclasses import dataclass

_VALID_SOURCES = frozenset({"auto", "dpop", "browser", "fixture"})
_VALID_PROVIDERS = frozenset({"anthropic", "openai", "vertex"})


@dataclass(frozen=True)
class Settings:
    mercari_source: str = "auto"
    min_interval: float = 1.0
    cache_ttl: float = 300.0
    request_timeout: float = 20.0
    llm_provider: str = "anthropic"
    llm_model: str = ""
    llm_max_tokens: int = 1024
    llm_timeout: float = 60.0
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    vertex_project_id: str = ""
    vertex_region: str = ""

    def __repr__(self) -> str:
        anthropic = "set" if self.anthropic_api_key else "unset"
        openai = "set" if self.openai_api_key else "unset"
        return (
            "Settings("
            f"mercari_source={self.mercari_source!r}, "
            f"min_interval={self.min_interval!r}, "
            f"cache_ttl={self.cache_ttl!r}, "
            f"request_timeout={self.request_timeout!r}, "
            f"llm_provider={self.llm_provider!r}, "
            f"llm_model={self.llm_model!r}, "
            f"llm_max_tokens={self.llm_max_tokens!r}, "
            f"llm_timeout={self.llm_timeout!r}, "
            f"anthropic_api_key={anthropic}, "
            f"openai_api_key={openai}, "
            f"vertex_project_id={self.vertex_project_id!r}, "
            f"vertex_region={self.vertex_region!r})"
        )


def load_settings() -> Settings:
    source = os.environ.get("MERCARI_SOURCE", "auto").strip().lower() or "auto"
    if source not in _VALID_SOURCES:
        raise ValueError(f"MERCARI_SOURCE={source!r} is not one of {sorted(_VALID_SOURCES)}")
    provider = os.environ.get("LLM_PROVIDER", "anthropic").strip().lower() or "anthropic"
    if provider not in _VALID_PROVIDERS:
        raise ValueError(f"LLM_PROVIDER={provider!r} is not one of {sorted(_VALID_PROVIDERS)}")
    max_tokens = _env_int("LLM_MAX_TOKENS", 1024)
    timeout = _env_float("LLM_TIMEOUT", 60.0)
    if max_tokens < 1:
        raise ValueError("LLM_MAX_TOKENS must be at least 1")
    if timeout <= 0:
        raise ValueError("LLM_TIMEOUT must be positive")
    return Settings(
        mercari_source=source,
        min_interval=_env_float("MERCARI_MIN_INTERVAL", 1.0),
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
