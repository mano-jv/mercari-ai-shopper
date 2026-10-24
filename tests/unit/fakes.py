from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from mercari_shopper.domain.models import Listing, SearchQuery, SearchResult
from mercari_shopper.llm.base import (
    LLMResponse,
    Message,
    ToolChoice,
    ToolChoiceMode,
    ToolSpec,
)


class FakeSource:
    def __init__(
        self,
        *,
        error: BaseException | None = None,
        result: SearchResult | None = None,
        items: list[Listing] | None = None,
    ) -> None:
        self.calls = 0
        self.item_calls = 0
        self.error = error
        self.result = result if result is not None else SearchResult(items=[])
        self.items = items or []
        self.queries: list[SearchQuery] = []

    async def search(self, query: SearchQuery) -> SearchResult:
        self.calls += 1
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return self.result

    async def get_items(self, item_ids: list[str]) -> list[Listing]:
        self.item_calls += 1
        if self.error is not None:
            raise self.error
        wanted = set(item_ids)
        return [item for item in self.items if item.item_id in wanted]


@dataclass
class CompleteRequest:
    messages: list[Message]
    tools: tuple[ToolSpec, ...]
    tool_choice: ToolChoice


class ScriptedLLM:
    """Returns canned LLMResponse values, or raises scripted exceptions."""

    def __init__(self, script: Sequence[LLMResponse | BaseException]) -> None:
        self._script = list(script)
        self.requests: list[CompleteRequest] = []

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = (),
        tool_choice: ToolChoice = ToolChoiceMode.AUTO,
    ) -> LLMResponse:
        self.requests.append(CompleteRequest(list(messages), tuple(tools), tool_choice))
        if not self._script:
            raise AssertionError("ScriptedLLM has no more responses")
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    async def aclose(self) -> None:
        return None
