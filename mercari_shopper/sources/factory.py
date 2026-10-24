"""Build a MercariSource from a name, wrapping cache and rate limiting."""

from __future__ import annotations

import httpx

from mercari_shopper.config import Settings, load_settings
from mercari_shopper.sources.api_source import DpopApiSource
from mercari_shopper.sources.auto_source import AutoSource
from mercari_shopper.sources.base import MercariSource
from mercari_shopper.sources.browser_source import BrowserSource, playwright_available
from mercari_shopper.sources.cache import CachedSource, TtlCache
from mercari_shopper.sources.fixture_source import FixtureSource
from mercari_shopper.sources.retry import RateLimiter


def build_source(
    kind: str | None = None,
    *,
    settings: Settings | None = None,
    client: httpx.AsyncClient | None = None,
    cache: TtlCache[object] | None = None,
    limiter: RateLimiter | None = None,
) -> CachedSource:
    resolved = settings or load_settings()
    name = (kind or resolved.mercari_source).strip().lower()
    rate_limiter = limiter or RateLimiter(min_interval=resolved.min_interval)
    inner: MercariSource
    if name == "fixture":
        inner = FixtureSource.from_package()
    elif name == "dpop":
        inner = DpopApiSource(client=client, limiter=rate_limiter, timeout=resolved.request_timeout)
    elif name == "browser":
        inner = BrowserSource(limiter=rate_limiter)
    elif name == "auto":
        primary = DpopApiSource(
            client=client, limiter=rate_limiter, timeout=resolved.request_timeout
        )
        fallback = BrowserSource(limiter=rate_limiter) if playwright_available() else None
        inner = AutoSource(primary, fallback)
    else:
        raise ValueError(f"unknown Mercari source {name!r}")
    return CachedSource(
        inner, cache if cache is not None else TtlCache(ttl_seconds=resolved.cache_ttl)
    )
