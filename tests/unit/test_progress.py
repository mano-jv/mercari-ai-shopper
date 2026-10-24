from mercari_shopper.agent.progress import STATUS_START, status_for_tools
from mercari_shopper.tools.schemas import (
    DETAIL_TOOL,
    MARKET_TOOL,
    SEARCH_TOOL,
    SUBMIT_TOOL,
)
from web.app import summary_stream_chunks


def test_status_for_tools_prefers_search_in_mixed_batch() -> None:
    assert status_for_tools((MARKET_TOOL, SEARCH_TOOL)) == "Searching Mercari…"


def test_status_for_tools_market_alone() -> None:
    assert status_for_tools((MARKET_TOOL,)) == "Comparing sold prices…"


def test_status_for_tools_submit() -> None:
    assert status_for_tools((SUBMIT_TOOL,)) == "Ranking your top picks…"


def test_status_for_tools_details() -> None:
    assert status_for_tools((DETAIL_TOOL,)) == "Fetching listing details…"


def test_status_for_tools_unknown() -> None:
    assert status_for_tools(("other",)) == "Working…"


def test_summary_stream_chunks_rebuilds_text() -> None:
    text = "Three picks under budget."
    assert "".join(summary_stream_chunks(text)) == text


def test_status_start_constant() -> None:
    assert "request" in STATUS_START.lower()
