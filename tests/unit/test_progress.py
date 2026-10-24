from mercari_shopper.agent.progress import (
    STATUS_START,
    ProgressTimeline,
    describe_tool_call,
    describe_tool_result,
    status_for_tools,
)
from mercari_shopper.llm.base import ToolCall
from mercari_shopper.tools.registry import ToolExecution
from mercari_shopper.tools.schemas import (
    ASK_TOOL,
    DETAIL_TOOL,
    MARKET_TOOL,
    SEARCH_TOOL,
    SUBMIT_TOOL,
)
from web.app import _render_timeline, summary_stream_chunks


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


def test_timeline_marks_previous_step_done() -> None:
    timeline = ProgressTimeline()
    first = timeline.advance("Searching Mercari…")
    assert first == timeline.steps
    assert first[0].text == "Searching Mercari…"
    assert first[0].done is False
    second = timeline.advance("Found 3 listings")
    assert second[0].done is True
    assert second[1].done is False
    done = timeline.finish()
    assert all(step.done for step in done)
    assert timeline.finish() == done


def test_timeline_finish_with_no_steps() -> None:
    assert ProgressTimeline().finish() == ()


def test_describe_tool_call_keeps_status_and_adds_args() -> None:
    calls = (
        ToolCall(
            "s",
            SEARCH_TOOL,
            {"keyword": "switch oled", "price_max": 30000, "limit": 10},
        ),
    )
    line = describe_tool_call(calls)
    assert "Searching Mercari" in line
    assert '"switch oled"' in line
    assert "≤¥30,000" in line


def test_describe_tool_call_uses_dominant_tool_in_mixed_batch() -> None:
    calls = (
        ToolCall("m", MARKET_TOOL, {"keyword": "switch"}),
        ToolCall("s", SEARCH_TOOL, {"keyword": "switch oled", "price_min": 10000}),
    )
    line = describe_tool_call(calls)
    assert line.startswith("Searching Mercari")
    assert "≥¥10,000" in line


def test_describe_tool_call_details_counts_ids() -> None:
    calls = (ToolCall("d", DETAIL_TOOL, {"item_ids": ["m1", "m2", "m3"]}),)
    assert describe_tool_call(calls) == "Fetching listing details… (3 items)"


def test_describe_tool_result_search_and_market_and_details() -> None:
    search = ToolExecution(
        "s",
        SEARCH_TOOL,
        "{}",
        False,
        data={"returned": 12, "num_found": 48, "items": [{}] * 12},
    )
    market = ToolExecution(
        "m",
        MARKET_TOOL,
        "{}",
        False,
        data={"sample_size": 23, "median": 24000},
    )
    details = ToolExecution(
        "d",
        DETAIL_TOOL,
        "{}",
        False,
        data={"items": [{}, {}, {}], "missing_ids": []},
    )
    assert describe_tool_result(search) == "Found 12 of 48 listings"
    assert describe_tool_result(market) == "23 sold listings · median ¥24,000"
    assert describe_tool_result(details) == "Fetched details for 3 items"


def test_describe_tool_result_skips_terminals_and_summarizes_errors() -> None:
    submit = ToolExecution("f", SUBMIT_TOOL, "{}", False, terminal=True, data={"summary": "x"})
    ask = ToolExecution("q", ASK_TOOL, "{}", False, terminal=True, data={"question": "What?"})
    failed = ToolExecution(
        "s",
        SEARCH_TOOL,
        '{"error":"tool failed","detail":"timeout"}',
        True,
    )
    assert describe_tool_result(submit) is None
    assert describe_tool_result(ask) is None
    assert describe_tool_result(failed) == "Searching Mercari failed: timeout"


def test_render_timeline_uses_done_and_pending_marks() -> None:
    timeline = ProgressTimeline()
    timeline.advance("Searching Mercari…")
    text = _render_timeline(timeline.advance("Found 3 listings"))
    assert "✅ Searching Mercari…" in text
    assert "⏳ Found 3 listings" in text
