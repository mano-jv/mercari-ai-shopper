"""Human-readable status lines for UIs during `ShopperAgent.run`."""

from __future__ import annotations

from collections.abc import Callable

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


def status_for_tools(tool_names: tuple[str, ...]) -> str:
    """Pick one status line for a batch of tool calls in a single turn."""
    names = set(tool_names)
    for name in _TOOL_PRIORITY:
        if name in names:
            return _TOOL_STATUS[name]
    return "Working…"


def emit_progress(callback: ProgressCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)
