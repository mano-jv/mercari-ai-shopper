from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from mercari_shopper.agent.errors import AgentError
from mercari_shopper.agent.loop import AgentResult
from mercari_shopper.domain.models import ItemStatus, Listing, SearchQuery, SearchResult
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
        sold: SearchResult | None = None,
    ) -> None:
        self.calls = 0
        self.item_calls = 0
        self.error = error
        self.result = result if result is not None else SearchResult(items=[])
        self.sold = sold
        self.items = items or []
        self.queries: list[SearchQuery] = []

    async def search(self, query: SearchQuery) -> SearchResult:
        self.calls += 1
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        if self.sold is not None and ItemStatus.SOLD_OUT in query.status:
            return self.sold
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


class ImmediateShopper:
    """Returns a canned AgentResult. Used by CLI and presenter tests."""

    def __init__(self, result: AgentResult) -> None:
        self.result = result
        self.requests: list[str] = []

    async def run(self, request: str, **kwargs: object) -> AgentResult:
        self.requests.append(request)
        return self.result


class ErrorShopper:
    def __init__(self, error: AgentError) -> None:
        self.error = error
        self.requests: list[str] = []

    async def run(self, request: str, **kwargs: object) -> AgentResult:
        self.requests.append(request)
        raise self.error
