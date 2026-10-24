"""Registry of tools the agent is allowed to call."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from mercari_shopper.llm.base import ToolSpec
from mercari_shopper.logging import log_event
from mercari_shopper.sources.base import MercariSource
from mercari_shopper.sources.errors import MercariError
from mercari_shopper.tools.compact import ResultCompactor
from mercari_shopper.tools.errors import ToolError, ToolValidationError, UnknownToolError
from mercari_shopper.tools.handlers import (
    GetItemDetailsTool,
    SearchMercariTool,
    SubmitRecommendationsTool,
    Tool,
)


class ToolCallLike(Protocol):
    """Enough of a tool call for the registry. Matches llm.ToolCall."""

    @property
    def id(self) -> str: ...

    @property
    def name(self) -> str: ...

    @property
    def arguments(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class ToolExecution:
    """What the agent appends as a tool result. `data` is the structured payload."""

    tool_call_id: str
    name: str
    content: str
    is_error: bool
    terminal: bool = False
    data: object | None = None


class ToolRegistry:
    def __init__(self, tools: Sequence[Tool]) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool {tool.name}")
            self._tools[tool.name] = tool

    @classmethod
    def for_source(
        cls,
        source: MercariSource,
        compactor: ResultCompactor | None = None,
    ) -> ToolRegistry:
        """Factory method: the three tools Phase 2 exposes, sharing one source."""
        shared = compactor or ResultCompactor()
        return cls(
            [
                SearchMercariTool(source, compactor=shared),
                GetItemDetailsTool(source, compactor=shared),
                SubmitRecommendationsTool(),
            ]
        )

    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(tool.spec() for tool in self._tools.values())

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise UnknownToolError(name, self.names()) from exc

    def is_terminal(self, name: str) -> bool:
        tool = self._tools.get(name)
        return tool is not None and tool.terminal

    async def execute(self, call: ToolCallLike) -> ToolExecution:
        tool = self._tools.get(call.name)
        if tool is None:
            return self._failure(
                call,
                {
                    "error": "unknown tool",
                    "tool": call.name,
                    "known_tools": list(self.names()),
                },
            )
        argument_error = getattr(call, "argument_error", None)
        if isinstance(argument_error, str) and argument_error:
            return self._failure(
                call,
                {"error": "invalid arguments", "tool": call.name, "detail": argument_error},
            )
        try:
            data = await tool.run(call.arguments)
        except ToolValidationError as exc:
            return self._failure(
                call,
                {"error": "invalid arguments", "tool": call.name, "detail": exc.detail},
            )
        except ToolError as exc:
            return self._failure(
                call, {"error": "tool failed", "tool": call.name, "detail": str(exc)}
            )
        except MercariError as exc:
            return self._failure(
                call,
                {"error": "mercari request failed", "tool": call.name, "detail": str(exc)},
            )
        execution = ToolExecution(
            tool_call_id=call.id,
            name=call.name,
            content=dump_json(data),
            is_error=False,
            terminal=tool.terminal,
            data=data,
        )
        log_event("tool_call", tool=call.name, error=False, terminal=execution.terminal)
        return execution

    async def execute_many(self, calls: Sequence[ToolCallLike]) -> list[ToolExecution]:
        """Run a batch. One failure becomes an error result and the others still finish."""
        return list(await asyncio.gather(*(self._isolated(call) for call in calls)))

    async def _isolated(self, call: ToolCallLike) -> ToolExecution:
        try:
            return await self.execute(call)
        except Exception as exc:
            return self._failure(
                call,
                {
                    "error": "tool failed",
                    "tool": call.name,
                    "detail": f"{type(exc).__name__}: {exc}",
                },
            )

    def _failure(self, call: ToolCallLike, payload: dict[str, Any]) -> ToolExecution:
        log_event("tool_call", tool=call.name, error=True, terminal=False)
        return ToolExecution(
            tool_call_id=call.id,
            name=call.name,
            content=dump_json(payload),
            is_error=True,
        )


def dump_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
