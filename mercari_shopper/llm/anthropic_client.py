"""Anthropic Messages API adapter.

Tool calls are `tool_use` blocks. Tool results go back in one user message,
because the API does not have a separate tool role. The system prompt is a
top-level parameter, not a message.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from anthropic import AsyncAnthropic

from mercari_shopper.llm.adapter import LLMAdapter
from mercari_shopper.llm.base import (
    LLMResponse,
    Message,
    Role,
    ToolCall,
    ToolChoice,
    ToolChoiceMode,
    ToolSpec,
    coerce_arguments,
)
from mercari_shopper.llm.errors import LLMError


class AnthropicAdapter(LLMAdapter):
    def __init__(
        self,
        *,
        model: str,
        api_key: str = "",
        max_tokens: int = 1024,
        timeout: float = 60.0,
        client: Any | None = None,
        owns_client: bool | None = None,
        provider: str = "anthropic",
    ) -> None:
        if owns_client is None:
            owns_client = client is None
        sdk = client if client is not None else AsyncAnthropic(api_key=api_key, timeout=timeout)
        super().__init__(
            provider=provider,
            model=model,
            max_tokens=max_tokens,
            timeout=timeout,
            client=sdk,
            owns_client=owns_client,
        )

    def build_payload(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec],
        tool_choice: ToolChoice,
    ) -> dict[str, Any]:
        system, converted = _to_messages(messages)
        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": converted,
            "timeout": self._timeout,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [_tool_param(tool) for tool in tools]
            payload["tool_choice"] = _tool_choice(tool_choice)
        return payload

    async def send(self, payload: dict[str, Any]) -> Any:
        return await self._client.messages.create(**payload)

    def parse(self, raw: Any) -> LLMResponse:
        content = getattr(raw, "content", None)
        if not isinstance(content, list):
            raise LLMError(f"{self.provider} response has no content list", provider=self.provider)
        texts: list[str] = []
        calls: list[ToolCall] = []
        for block in content:
            block_type = getattr(block, "type", "")
            if block_type == "text":
                texts.append(str(getattr(block, "text", "")))
            elif block_type == "tool_use":
                arguments, argument_error = coerce_arguments(getattr(block, "input", {}))
                calls.append(
                    ToolCall(
                        id=str(getattr(block, "id", "")),
                        name=str(getattr(block, "name", "")),
                        arguments=arguments,
                        argument_error=argument_error,
                    )
                )
        stop = getattr(raw, "stop_reason", None)
        return LLMResponse(
            message=Message(
                role=Role.ASSISTANT,
                content="".join(texts),
                tool_calls=tuple(calls),
            ),
            stop_reason="" if stop is None else str(stop),
        )


def _to_messages(messages: Sequence[Message]) -> tuple[str | None, list[dict[str, Any]]]:
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    def flush() -> None:
        if not pending:
            return
        _append_user(converted, list(pending))
        pending.clear()

    for message in messages:
        if message.role is Role.SYSTEM:
            flush()
            if message.content:
                system_parts.append(message.content)
            continue
        if message.role is Role.TOOL:
            pending.append(_tool_result_block(message))
            continue
        flush()
        if message.role is Role.USER:
            _append_user(converted, message.content)
            continue
        converted.append({"role": "assistant", "content": _assistant_blocks(message)})
    flush()
    system = "\n\n".join(system_parts) if system_parts else None
    return system, converted


def _append_user(converted: list[dict[str, Any]], content: str | list[dict[str, Any]]) -> None:
    """Fold consecutive user turns into one. Anthropic rejects back-to-back user roles."""
    if not converted or converted[-1]["role"] != "user":
        converted.append({"role": "user", "content": content})
        return
    previous = converted[-1]["content"]
    if isinstance(previous, str) and isinstance(content, str):
        converted[-1]["content"] = f"{previous}\n{content}" if previous else content
        return
    blocks = _as_user_blocks(previous)
    if isinstance(content, str):
        if content:
            blocks.append({"type": "text", "text": content})
    else:
        blocks.extend(content)
    converted[-1]["content"] = blocks


def _as_user_blocks(content: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(content, list):
        return list(content)
    if not content:
        return []
    return [{"type": "text", "text": content}]


def _assistant_blocks(message: Message) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if message.content:
        blocks.append({"type": "text", "text": message.content})
    for call in message.tool_calls:
        blocks.append(
            {
                "type": "tool_use",
                "id": call.id,
                "name": call.name,
                "input": dict(call.arguments),
            }
        )
    if not blocks:
        raise ValueError("assistant message has no text and no tool calls")
    return blocks


def _tool_result_block(message: Message) -> dict[str, Any]:
    result = message.tool_result
    if result is None:
        raise ValueError("tool message is missing tool_result")
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": result.tool_call_id,
        "content": result.content,
    }
    if result.is_error:
        block["is_error"] = True
    return block


def _tool_param(tool: ToolSpec) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": dict(tool.parameters),
    }


def _tool_choice(choice: ToolChoice) -> dict[str, str]:
    if isinstance(choice, ToolChoiceMode):
        return {"type": choice.value}
    return {"type": "tool", "name": choice.name}
