"""Direct DPoP-signed calls to api.mercari.jp."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Awaitable, Callable

import httpx

from mercari_shopper.domain.models import Listing, SearchQuery, SearchResult
from mercari_shopper.logging import log_event
from mercari_shopper.sources.base import SourceLifecycle, unique_ids
from mercari_shopper.sources.dpop import DpopSigner
from mercari_shopper.sources.errors import (
    MercariHttpError,
    MercariParseError,
    MercariTransportError,
    parse_retry_after,
)
from mercari_shopper.sources.parser import parse_item_response, parse_search_response
from mercari_shopper.sources.payload import (
    DEFAULT_HEADERS,
    SEARCH_URL,
    build_search_body,
    item_request_url,
)
from mercari_shopper.sources.retry import RateLimiter, retry_async


class DpopApiSource(SourceLifecycle):
    """Fast path: sign each request with an ephemeral P-256 key."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        signer: DpopSigner | None = None,
        limiter: RateLimiter | None = None,
        timeout: float = 20.0,
        retry_attempts: int = 3,
        retry_sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._client = client
        self._owns_client = client is None
        self._signer = signer or DpopSigner()
        self._limiter = limiter or RateLimiter()
        self._timeout = timeout
        self._retry_attempts = retry_attempts
        self._retry_sleep = retry_sleep

    async def search(self, query: SearchQuery) -> SearchResult:
        async def once() -> SearchResult:
            body = build_search_body(
                query,
                session_id=uuid.uuid4().hex,
                device_uuid=self._signer.device_uuid,
            )
            payload = await self._request("POST", SEARCH_URL, json_body=body)
            result = parse_search_response(payload)
            log_event(
                "mercari_search",
                source="dpop",
                keyword=query.keyword,
                n=len(result.items),
                num_found=result.num_found,
            )
            return result

        return await self._with_retry(once)

    async def get_items(self, item_ids: list[str]) -> list[Listing]:
        semaphore = asyncio.Semaphore(self._limiter.max_concurrent)

        async def bounded(item_id: str) -> Listing | None:
            async with semaphore:
                return await self._get_item(item_id)

        results = await asyncio.gather(*(bounded(item_id) for item_id in unique_ids(item_ids)))
        return [listing for listing in results if listing is not None]

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _get_item(self, item_id: str) -> Listing | None:
        async def once() -> Listing | None:
            url = item_request_url(item_id)
            try:
                payload = await self._request("GET", url)
            except MercariHttpError as error:
                if error.status == 404:
                    return None
                raise
            return parse_item_response(payload)

        return await self._with_retry(once)

    async def _with_retry[T](self, operation: Callable[[], Awaitable[T]]) -> T:
        if self._retry_sleep is None:
            return await retry_async(operation, attempts=self._retry_attempts)
        return await retry_async(operation, attempts=self._retry_attempts, sleep=self._retry_sleep)

    async def _request(
        self, method: str, url: str, *, json_body: dict[str, object] | None = None
    ) -> object:
        await self._limiter.wait()
        client = await self._http()
        headers = {**DEFAULT_HEADERS, "DPoP": self._signer.proof(url, method)}
        try:
            response = await client.request(
                method,
                url,
                headers=headers,
                json=json_body,
                timeout=self._timeout,
            )
        except httpx.RequestError as error:
            raise MercariTransportError(str(error)) from error
        if response.status_code >= 400:
            retry_after = parse_retry_after(response.headers.get("Retry-After"))
            raise MercariHttpError(
                response.status_code,
                response.text[:300],
                retry_after=retry_after,
            )
        try:
            return response.json()
        except json.JSONDecodeError as error:
            raise MercariParseError("response is not valid JSON") from error

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(headers=DEFAULT_HEADERS, timeout=self._timeout)
        return self._client
