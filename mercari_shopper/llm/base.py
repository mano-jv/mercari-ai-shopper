"""Normalized chat types. Provider adapters translate these; the agent does not."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolChoiceMode(StrEnum):
    """auto: the model may call a tool. any: it must. none: it must not."""

    AUTO = "auto"
    ANY = "any"
    NONE = "none"


@dataclass(frozen=True)
class ForcedTool:
    """Ask the provider to call this tool and no other."""

    name: str


type ToolChoice = ToolChoiceMode | ForcedTool


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: Mapping[str, Any]
    argument_error: str | None = None


@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class Message:
    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_result: ToolResult | None = None


@dataclass(frozen=True)
class ToolSpec:
    """Provider-neutral tool definition. `parameters` is a JSON Schema object."""

    name: str
    description: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class LLMResponse:
    message: Message
    stop_reason: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def text(self) -> str:
        return self.message.content

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        return self.message.tool_calls


@runtime_checkable
class LLMClient(Protocol):
    """Port the agent loop calls. One implementation per provider."""

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = (),
        tool_choice: ToolChoice = ToolChoiceMode.AUTO,
    ) -> LLMResponse:
        """Send the transcript and return the assistant turn."""

    async def aclose(self) -> None:
        """Release the underlying SDK client when this adapter owns it."""


def check_transcript(messages: Sequence[Message]) -> None:
    """Reject transcripts the adapters cannot represent."""
    for message in messages:
        if message.role is Role.TOOL and message.tool_result is None:
            raise ValueError("tool message is missing tool_result")
        if message.role is not Role.TOOL and message.tool_result is not None:
            raise ValueError(f"{message.role.value} message cannot include a tool result")
        if message.role is not Role.ASSISTANT and message.tool_calls:
            raise ValueError(f"{message.role.value} message cannot include tool calls")


def guard_tool_choice(choice: ToolChoice, tools: Sequence[ToolSpec]) -> None:
    names = {tool.name for tool in tools}
    if isinstance(choice, ForcedTool):
        if choice.name not in names:
            raise ValueError(f"forced tool {choice.name!r} is not in the tool list")
        return
    if choice is ToolChoiceMode.ANY and not names:
        raise ValueError("tool_choice any requires at least one tool")


def coerce_arguments(value: object) -> tuple[dict[str, Any], str | None]:
    """Return a JSON object and, when the model sent unusable arguments, an error.

    An empty string is an empty object. Invalid JSON and non-objects keep the
    error so the registry can hand it back to the model instead of guessing.
    """
    parsed = value
    if isinstance(parsed, str):
        if not parsed.strip():
            return {}, None
        try:
            parsed = json.loads(parsed)
        except json.JSONDecodeError as exc:
            return {}, f"tool arguments are not valid JSON: {exc.msg}"
    if not isinstance(parsed, dict):
        return {}, "tool arguments must be a JSON object"
    return {str(key): item for key, item in parsed.items()}, None
