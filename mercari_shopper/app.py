"""Composition root: settings + factories become a Shopper the CLI can run."""

from __future__ import annotations

import sys

from mercari_shopper.agent.enrich import build_enricher
from mercari_shopper.agent.factory import build_agent
from mercari_shopper.agent.loop import AgentResult
from mercari_shopper.agent.progress import ProgressCallback
from mercari_shopper.config import Settings, load_settings
from mercari_shopper.llm.base import LLMClient
from mercari_shopper.llm.factory import build_llm
from mercari_shopper.sources.cache import CachedSource
from mercari_shopper.sources.factory import build_source
from mercari_shopper.tools.registry import ToolRegistry


class LiveShopper:
    """Facade: owns the LLM and Mercari source for one or more requests."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        source_kind: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        llm: LLMClient | None = None,
        allow_clarify: bool = False,
    ) -> None:
        self._settings = settings or load_settings()
        self._source_kind = source_kind
        self._provider = provider
        self._model = model
        self._injected_llm = llm
        self._allow_clarify = allow_clarify
        self._llm: LLMClient | None = None
        self._source: CachedSource | None = None

    async def __aenter__(self) -> LiveShopper:
        llm = self._injected_llm or build_llm(
            self._provider, settings=self._settings, model=self._model
        )
        source: CachedSource | None = None
        try:
            source = build_source(self._source_kind, settings=self._settings)
            await source.__aenter__()
            self._llm = llm
            self._source = source
            warm = getattr(llm, "warm", None)
            if callable(warm):
                await warm()
            return self
        except BaseException:
            if source is not None:
                await source.__aexit__(*sys.exc_info())
            await self._close_owned_llm(llm)
            raise

    async def run(
        self,
        request: str,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> AgentResult:
        if self._llm is None or self._source is None:
            raise RuntimeError("LiveShopper must be used as an async context manager")
        # Fresh agent per request so grounding state never leaks across searches
        # while the LLM HTTP client and Mercari cache stay warm.
        agent = build_agent(
            self._llm,
            ToolRegistry.for_source(self._source, allow_clarify=self._allow_clarify),
            settings=self._settings,
            enricher=build_enricher(self._source),
            allow_clarify=self._allow_clarify,
        )
        return await agent.run(request, on_progress=on_progress)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        try:
            await self._close_owned_llm(self._llm)
        finally:
            source = self._source
            if source is not None:
                await source.__aexit__(exc_type, exc, tb)
            self._llm = None
            self._source = None

    async def _close_owned_llm(self, llm: LLMClient | None) -> None:
        if self._injected_llm is not None or llm is None:
            return
        await llm.aclose()
