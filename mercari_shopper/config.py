"""Runtime settings. Secrets stay in the environment, never in files."""

from __future__ import annotations

import os
from dataclasses import dataclass

_VALID_SOURCES = frozenset({"auto", "dpop", "browser", "fixture"})


@dataclass(frozen=True)
class Settings:
    mercari_source: str = "auto"
    min_interval: float = 1.0
    cache_ttl: float = 300.0
    request_timeout: float = 20.0


def load_settings() -> Settings:
    source = os.environ.get("MERCARI_SOURCE", "auto").strip().lower() or "auto"
    if source not in _VALID_SOURCES:
        raise ValueError(f"MERCARI_SOURCE={source!r} is not one of {sorted(_VALID_SOURCES)}")
    return Settings(
        mercari_source=source,
        min_interval=_env_float("MERCARI_MIN_INTERVAL", 1.0),
        cache_ttl=_env_float("MERCARI_CACHE_TTL", 300.0),
        request_timeout=_env_float("MERCARI_TIMEOUT", 20.0),
    )


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)
