"""Human-readable status lines for UIs during `ShopperAgent.run`."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from mercari_shopper.tools.schemas import (
    ASK_TOOL,
    DETAIL_TOOL,
    MARKET_TOOL,
    SEARCH_TOOL,
    SUBMIT_TOOL,
)

ProgressCallback = Callable[[str], None]

STATUS_START = "Understanding your request…"
STATUS_PLANNING = "Planning next step…"
STATUS_ENRICHING = "Enriching seller ratings…"

_TOOL_STATUS: dict[str, str] = {
    SEARCH_TOOL: "Searching Mercari…",
    MARKET_TOOL: "Comparing sold prices…",
    DETAIL_TOOL: "Fetching listing details…",
    ASK_TOOL: "Need a quick clarification…",
    SUBMIT_TOOL: "Ranking your top picks…",
}

# When several tools run in one turn, show the earliest phase in the pipeline.
_TOOL_PRIORITY: tuple[str, ...] = (
    SEARCH_TOOL,
    MARKET_TOOL,
    DETAIL_TOOL,
    SUBMIT_TOOL,
    ASK_TOOL,
)

_KEYWORD_LIMIT = 40


class ToolCallLike(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def arguments(self) -> Mapping[str, Any]: ...


class ToolOutcome(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def is_error(self) -> bool: ...

    @property
    def data(self) -> object: ...

    @property
    def content(self) -> str: ...


@dataclass(frozen=True)
class ProgressStep:
    """One line in a live activity timeline."""

    text: str
    done: bool = False


class ProgressTimeline:
    """Turn a stream of status strings into a growing checklist.

    The previous step is marked done when a new one starts. `finish` closes
    the last step so UIs can freeze the list when the run ends.
    """

    def __init__(self) -> None:
        self._steps: list[ProgressStep] = []

    @property
    def steps(self) -> tuple[ProgressStep, ...]:
        return tuple(self._steps)

    def advance(self, message: str) -> tuple[ProgressStep, ...]:
        self._close_last()
        self._steps.append(ProgressStep(message, done=False))
        return self.steps

    def finish(self) -> tuple[ProgressStep, ...]:
        self._close_last()
        return self.steps

    def _close_last(self) -> None:
        if self._steps and not self._steps[-1].done:
            last = self._steps[-1]
            self._steps[-1] = ProgressStep(last.text, done=True)


def status_for_tools(tool_names: tuple[str, ...]) -> str:
    """Pick one status line for a batch of tool calls in a single turn."""
    names = set(tool_names)
    for name in _TOOL_PRIORITY:
        if name in names:
            return _TOOL_STATUS[name]
    return "Working…"


def describe_tool_call(calls: Sequence[ToolCallLike]) -> str:
    """Status line plus a short argument summary for the dominant tool."""
    names = tuple(call.name for call in calls)
    status = status_for_tools(names)
    dominant = _dominant_name(names)
    if dominant is None:
        return status
    call = next((item for item in calls if item.name == dominant), None)
    if call is None:
        return status
    extra = _argument_summary(dominant, call.arguments)
    if not extra:
        return status
    return f"{status} ({extra})"


def describe_tool_result(execution: ToolOutcome) -> str | None:
    """Short 'done' line after a search/market/detail call. None for terminals."""
    if execution.name in (SUBMIT_TOOL, ASK_TOOL):
        return None
    payload = _payload(execution)
    if execution.is_error:
        return _error_summary(execution.name, payload)
    if execution.name == SEARCH_TOOL:
        return _search_result(payload)
    if execution.name == MARKET_TOOL:
        return _market_result(payload)
    if execution.name == DETAIL_TOOL:
        return _detail_result(payload)
    return None


def emit_progress(callback: ProgressCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _dominant_name(tool_names: tuple[str, ...]) -> str | None:
    names = set(tool_names)
    for name in _TOOL_PRIORITY:
        if name in names:
            return name
    return tool_names[0] if tool_names else None


def _argument_summary(name: str, arguments: Mapping[str, Any]) -> str:
    parts: list[str] = []
    if name in (SEARCH_TOOL, MARKET_TOOL):
        keyword = arguments.get("keyword")
        if isinstance(keyword, str) and keyword.strip():
            parts.append(f'"{_truncate(keyword.strip(), _KEYWORD_LIMIT)}"')
    if name == SEARCH_TOOL:
        price_min = arguments.get("price_min")
        if _is_int(price_min):
            parts.append(f"≥¥{price_min:,}")
        price_max = arguments.get("price_max")
        if _is_int(price_max):
            parts.append(f"≤¥{price_max:,}")
    if name == DETAIL_TOOL:
        ids = arguments.get("item_ids")
        if isinstance(ids, list) and ids:
            noun = "item" if len(ids) == 1 else "items"
            parts.append(f"{len(ids)} {noun}")
    return ", ".join(parts)


def _search_result(payload: dict[str, Any]) -> str | None:
    returned = payload.get("returned")
    found = payload.get("num_found")
    if _is_int(returned) and _is_int(found):
        return f"Found {returned} of {found} listings"
    if _is_int(returned):
        return f"Found {returned} listings"
    items = payload.get("items")
    if isinstance(items, list):
        return f"Found {len(items)} listings"
    return None


def _market_result(payload: dict[str, Any]) -> str | None:
    sample = payload.get("sample_size")
    median = payload.get("median")
    if _is_int(sample) and _is_int(median):
        return f"{sample} sold listings · median ¥{median:,}"
    if _is_int(sample):
        return f"{sample} sold listings"
    return None


def _detail_result(payload: dict[str, Any]) -> str | None:
    items = payload.get("items")
    if not isinstance(items, list):
        return None
    noun = "item" if len(items) == 1 else "items"
    return f"Fetched details for {len(items)} {noun}"


def _error_summary(name: str, payload: dict[str, Any]) -> str:
    label = _TOOL_STATUS.get(name, name).rstrip("…")
    detail = payload.get("detail") or payload.get("error")
    if isinstance(detail, str) and detail.strip():
        return f"{label} failed: {detail.strip()}"
    return f"{label} failed"


def _payload(execution: ToolOutcome) -> dict[str, Any]:
    data = execution.data
    if isinstance(data, dict):
        return data
    try:
        parsed = json.loads(execution.content)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit <= 1:
        return text[:limit]
    return text[: limit - 1] + "…"
