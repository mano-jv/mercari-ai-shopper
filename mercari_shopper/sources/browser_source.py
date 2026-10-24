"""Playwright interception of the search JSON the public page already fetches."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from mercari_shopper.domain.models import Listing, SearchQuery, SearchResult, item_url
from mercari_shopper.logging import log_event
from mercari_shopper.sources.base import SourceLifecycle, unique_ids
from mercari_shopper.sources.errors import MercariError, MercariHttpError, MercariTransportError
from mercari_shopper.sources.parser import parse_item_response, parse_search_response
from mercari_shopper.sources.payload import USER_AGENT, build_search_page_url
from mercari_shopper.sources.retry import RateLimiter

InterceptFn = Callable[[str, str], Awaitable[tuple[int, object]]]


def playwright_available() -> bool:
    try:
        import playwright.async_api  # noqa: F401
    except ImportError:
        return False
    return True


class BrowserSource(SourceLifecycle):
    """Robust path: let Mercari's own JavaScript sign DPoP, then read the response."""

    def __init__(
        self,
        *,
        limiter: RateLimiter | None = None,
        intercept: InterceptFn | None = None,
        timeout_ms: float = 25_000,
    ) -> None:
        self._limiter = limiter or RateLimiter()
        self._intercept = intercept
        self._timeout_ms = timeout_ms

    async def search(self, query: SearchQuery) -> SearchResult:
        page_url = build_search_page_url(query)
        status, payload = await self._fetch(page_url, "entities:search")
        if status >= 400:
            raise MercariHttpError(status, "browser intercepted an HTTP error from search")
        result = parse_search_response(payload)
        if len(result.items) > query.page_size:
            result = result.model_copy(update={"items": result.items[: query.page_size]})
        log_event(
            "mercari_search",
            source="browser",
            keyword=query.keyword,
            n=len(result.items),
            num_found=result.num_found,
        )
        return result

    async def get_items(self, item_ids: list[str]) -> list[Listing]:
        listings: list[Listing] = []
        for item_id in unique_ids(item_ids):
            page_url = item_url(item_id, is_shop_item=not item_id.startswith("m"))
            try:
                status, payload = await self._fetch(page_url, "items/get")
            except MercariTransportError:
                continue
            if status == 404:
                continue
            if status >= 400:
                raise MercariHttpError(status, f"browser intercepted HTTP {status} for {item_id}")
            listings.append(parse_item_response(payload))
        return listings

    async def _fetch(self, page_url: str, url_fragment: str) -> tuple[int, object]:
        await self._limiter.wait()
        intercept = self._intercept or self._playwright_intercept
        return await intercept(page_url, url_fragment)

    async def _playwright_intercept(self, page_url: str, url_fragment: str) -> tuple[int, object]:
        try:
            from playwright.async_api import TimeoutError as PlaywrightTimeoutError
            from playwright.async_api import async_playwright
        except ImportError as error:
            raise MercariError(
                "Playwright is not installed. "
                "Run: pip install '.[browser]' && playwright install chromium"
            ) from error

        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            try:
                context = await browser.new_context(locale="ja-JP", user_agent=USER_AGENT)
                page = await context.new_page()
                try:
                    async with page.expect_response(
                        lambda response: url_fragment in response.url,
                        timeout=self._timeout_ms,
                    ) as pending:
                        await page.goto(page_url, wait_until="domcontentloaded")
                    response = await pending.value
                except PlaywrightTimeoutError as error:
                    raise MercariTransportError(
                        f"timed out waiting for {url_fragment} on {page_url}"
                    ) from error
                status = response.status
                try:
                    payload: Any = await response.json()
                except Exception as error:
                    raise MercariTransportError("intercepted response was not JSON") from error
                return status, payload
            finally:
                await browser.close()
