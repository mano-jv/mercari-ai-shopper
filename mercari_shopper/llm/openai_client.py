"""OpenAI Chat Completions adapter.

Tool calls travel as `tool_calls` on the assistant message, with arguments
stored as a JSON string. Results are separate messages with role `tool`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from openai import AsyncOpenAI

from mercari_shopper.config import DEFAULT_LLM_MAX_TOKENS
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

_REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5")


def _is_reasoning_model(model: str) -> bool:
    lowered = model.lower()
    return lowered.startswith(_REASONING_MODEL_PREFIXES)


class OpenAIAdapter(LLMAdapter):
    def __init__(
        self,
        *,
        model: str,
        api_key: str = "",
        max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
        timeout: float = 60.0,
        client: Any | None = None,
        effort: str = "",
    ) -> None:
        owns_client = client is None
        sdk = client if client is not None else AsyncOpenAI(api_key=api_key, timeout=timeout)
        super().__init__(
            provider="openai",
            model=model,
            max_tokens=max_tokens,
            timeout=timeout,
            client=sdk,
            owns_client=owns_client,
            effort=effort,
        )

    def build_payload(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec],
        tool_choice: ToolChoice,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [_to_message(message) for message in messages],
            "max_completion_tokens": self._max_tokens,
            "timeout": self._timeout,
        }
        if tools:
            payload["tools"] = [_tool_param(tool) for tool in tools]
            payload["tool_choice"] = _tool_choice(tool_choice)
        if self._effort and _is_reasoning_model(self._model):
            payload["reasoning_effort"] = self._effort
        return payload

    async def send(self, payload: dict[str, Any]) -> Any:
        return await self._client.chat.completions.create(**payload)

    def parse(self, raw: Any) -> LLMResponse:
        choices = getattr(raw, "choices", None)
        if not isinstance(choices, list) or not choices:
            raise LLMError("openai response has no choices", provider=self.provider)
        choice = choices[0]
        message = getattr(choice, "message", None)
        if message is None:
            raise LLMError("openai response is missing the message", provider=self.provider)
        content = getattr(message, "content", None)
        text = content if isinstance(content, str) else ""
        calls: list[ToolCall] = []
        for call in getattr(message, "tool_calls", None) or []:
            function = getattr(call, "function", None)
            if function is None:
                raise LLMError("openai tool call is missing the function", provider=self.provider)
            raw_arguments = getattr(function, "arguments", "") or ""
            arguments, argument_error = coerce_arguments(raw_arguments)
            calls.append(
                ToolCall(
                    id=str(getattr(call, "id", "")),
                    name=str(getattr(function, "name", "")),
                    arguments=arguments,
                    argument_error=argument_error,
                )
            )
        finish = getattr(choice, "finish_reason", None)
        return LLMResponse(
            message=Message(role=Role.ASSISTANT, content=text, tool_calls=tuple(calls)),
            stop_reason="" if finish is None else str(finish),
        )


def _to_message(message: Message) -> dict[str, Any]:
    if message.role is Role.SYSTEM:
        return {"role": "system", "content": message.content}
    if message.role is Role.USER:
        return {"role": "user", "content": message.content}
    if message.role is Role.TOOL:
        result = message.tool_result
        if result is None:
            raise ValueError("tool message is missing tool_result")
        return {
            "role": "tool",
            "tool_call_id": result.tool_call_id,
            "content": result.content,
        }
    return _assistant_message(message)


def _assistant_message(message: Message) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": "assistant",
        "content": message.content if message.content else None,
    }
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(dict(call.arguments), ensure_ascii=False),
                },
            }
            for call in message.tool_calls
        ]
        return payload
    if not message.content:
        raise ValueError("assistant message has no text and no tool calls")
    return payload


def _tool_param(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": dict(tool.parameters),
        },
    }


def _tool_choice(choice: ToolChoice) -> str | dict[str, Any]:
    if isinstance(choice, ToolChoiceMode):
        if choice is ToolChoiceMode.ANY:
            return "required"
        return choice.value
    return {"type": "function", "function": {"name": choice.name}}
