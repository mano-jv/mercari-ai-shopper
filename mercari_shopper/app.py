"""Composition root: settings + factories become a Shopper the CLI can run."""

from __future__ import annotations

from mercari_shopper.agent.factory import build_agent
from mercari_shopper.agent.loop import AgentResult, ShopperAgent
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
    ) -> None:
        self._settings = settings or load_settings()
        self._source_kind = source_kind
        self._provider = provider
        self._model = model
        self._injected_llm = llm
        self._llm: LLMClient | None = None
        self._source: CachedSource | None = None
        self._agent: ShopperAgent | None = None

    async def __aenter__(self) -> LiveShopper:
        self._llm = self._injected_llm or build_llm(
            self._provider, settings=self._settings, model=self._model
        )
        self._source = build_source(self._source_kind, settings=self._settings)
        await self._source.__aenter__()
        self._agent = build_agent(
            self._llm,
            ToolRegistry.for_source(self._source),
            settings=self._settings,
        )
        return self

    async def run(self, request: str) -> AgentResult:
        if self._agent is None:
            raise RuntimeError("LiveShopper must be used as an async context manager")
        return await self._agent.run(request)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        if self._injected_llm is None and self._llm is not None:
            await self._llm.aclose()
        if self._source is not None:
            await self._source.__aexit__(exc_type, exc, tb)
        self._agent = None
        self._llm = None
        self._source = None
