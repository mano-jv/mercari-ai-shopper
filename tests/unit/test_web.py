from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest
from mercari_shopper.agent.errors import AgentBudgetError
from mercari_shopper.agent.loop import AgentResult
from mercari_shopper.agent.prompts import PromptBuilder
from mercari_shopper.config import Settings
from mercari_shopper.domain.models import Recommendation
from mercari_shopper.llm.base import ToolCall
from mercari_shopper.presentation.view import ResultViewBuilder, SellerView, format_yen
from mercari_shopper.tools.schemas import ASK_TOOL, SEARCH_TOOL, SUBMIT_TOOL
from tests.unit.fakes import ImmediateShopper, ScriptedLLM
from tests.unit.test_app import _FIXTURE_IDS, _rec, _tools
from tests.unit.test_presenter import _result
from web.app import (
    ShopperHolder,
    needs_clarification,
    next_request,
    optional_override,
    run_live,
    run_search,
    seller_line,
    source_kind,
)

ROOT = Path(__file__).resolve().parents[2]
APP_PATH = ROOT / "web" / "app.py"


def _app_test():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(APP_PATH), default_timeout=10)


def test_source_kind_offline_uses_fixtures() -> None:
    assert source_kind(True) == "fixture"
    assert source_kind(False) is None


def test_optional_override_treats_env_default_as_unset() -> None:
    assert optional_override(None) is None
    assert optional_override("  ") is None
    assert optional_override("(env default)") is None
    assert optional_override("vertex") == "vertex"
    assert optional_override("  claude-haiku-5-5  ") == "claude-haiku-5-5"


def test_next_request_wraps_clarification() -> None:
    wrapped = next_request("a nice gift", "Switch OLED")
    expected = PromptBuilder().with_clarification("a nice gift", "Switch OLED")
    assert wrapped == expected
    assert "Switch OLED" in wrapped
    assert ASK_TOOL in wrapped


def test_needs_clarification_only_when_question_without_recs() -> None:
    question = AgentResult(
        summary="",
        recommendations=(),
        listings=(),
        steps=1,
        tool_calls=1,
        repairs=0,
        question="What kind of gift?",
    )
    assert needs_clarification(question) is True
    assert needs_clarification(_result()) is False


def test_seller_line_and_yen_match_view_models() -> None:
    assert seller_line(None) == ""
    assert seller_line(SellerView(name="sample-seller", star_rating=5.0, num_ratings=122)) == (
        "sample-seller ★5 (122)"
    )
    view = ResultViewBuilder().build(_result())
    first = view.items[0]
    assert first.name == "Nintendo Switch 有機EL 本体"
    assert format_yen(first.price) == "¥24,800"
    assert first.condition == "no noticeable scratches"
    assert first.url.startswith("https://jp.mercari.com/")
    assert seller_line(first.seller) == "sample-seller ★5 (122)"


async def test_run_live_fixture_source_returns_top_three() -> None:
    llm = ScriptedLLM(
        [
            _tools(
                ToolCall("s", SEARCH_TOOL, {"keyword": "switch", "price_max": 30000, "limit": 10})
            ),
            _tools(
                ToolCall(
                    "f",
                    SUBMIT_TOOL,
                    {
                        "summary": "OLED Switch options under 30000 yen",
                        "recommendations": [
                            _rec(_FIXTURE_IDS[0], 1),
                            _rec(_FIXTURE_IDS[1], 2),
                            _rec(_FIXTURE_IDS[2], 3),
                        ],
                    },
                )
            ),
        ]
    )
    settings = Settings(mercari_source="fixture", anthropic_api_key="test")
    result = await run_live(
        "a used Nintendo Switch OLED under 30000 yen",
        offline=True,
        llm=llm,
        settings=settings,
    )
    view = ResultViewBuilder().build(result)
    assert [item.item_id for item in view.items] == list(_FIXTURE_IDS)
    assert view.items[0].name
    assert view.items[0].price is not None
    assert all(item.url for item in view.items)


async def test_run_live_clarify_then_follow_up() -> None:
    llm = ScriptedLLM(
        [
            _tools(ToolCall("q", ASK_TOOL, {"question": "What kind of gift?"})),
            _tools(ToolCall("s", SEARCH_TOOL, {"keyword": "switch", "limit": 10})),
            _tools(
                ToolCall(
                    "f",
                    SUBMIT_TOOL,
                    {
                        "summary": "Switch gift options",
                        "recommendations": [
                            _rec(_FIXTURE_IDS[0], 1),
                            _rec(_FIXTURE_IDS[1], 2),
                            _rec(_FIXTURE_IDS[2], 3),
                        ],
                    },
                )
            ),
        ]
    )
    settings = Settings(mercari_source="fixture", anthropic_api_key="test")
    first = await run_live("a nice gift", offline=True, llm=llm, settings=settings)
    assert needs_clarification(first) is True
    assert first.question == "What kind of gift?"

    second = await run_live(
        next_request("a nice gift", "Nintendo Switch OLED"),
        offline=True,
        llm=llm,
        settings=settings,
    )
    assert needs_clarification(second) is False
    assert [rec.item_id for rec in second.recommendations] == list(_FIXTURE_IDS)


def test_shopper_holder_reuses_one_live_shopper(monkeypatch: pytest.MonkeyPatch) -> None:
    canned = _result()
    created: list[Any] = []

    class _TrackingLive:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.runs = 0
            self.last_request = ""

        async def __aenter__(self) -> _TrackingLive:
            created.append(self)
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def run(self, request: str, **kwargs: object) -> AgentResult:
            del kwargs
            self.runs += 1
            self.last_request = request
            return canned

    monkeypatch.setattr("web.app.LiveShopper", _TrackingLive)
    holder = ShopperHolder()
    try:
        first = holder.search("find a switch", offline=True, provider="anthropic")
        second = holder.search("find a ps5", offline=True, provider="anthropic")
        openai = holder.search("find a switch", offline=True, provider="openai")
    finally:
        holder.close()
    assert first is canned
    assert second is canned
    assert openai is canned
    assert len(created) == 2
    assert created[0].kwargs["provider"] == "anthropic"
    assert created[0].runs == 2
    assert created[0].last_request == "find a ps5"
    assert created[1].kwargs["provider"] == "openai"
    assert created[1].runs == 1


def test_run_search_is_sync_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    canned = _result()

    async def fake_run(request: str, **kwargs: object) -> AgentResult:
        assert request == "find a switch"
        assert kwargs["offline"] is True
        assert kwargs["provider"] == "vertex"
        return canned

    monkeypatch.setattr("web.app.run_live", fake_run)
    result = run_search("find a switch", offline=True, provider="vertex")
    assert result is canned


def test_docker_and_compose_keep_secrets_out_of_the_image() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    ignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "streamlit" in dockerfile
    assert "web/app.py" in dockerfile
    assert ".env" in ignore
    assert "env_file: .env" in compose
    assert "ANTHROPIC_API_KEY=" not in dockerfile
    assert "OPENAI_API_KEY=" not in dockerfile


def test_pyproject_declares_streamlit_web_extra() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'web = ["streamlit>=1.40"]' in text


class _ErrorShopper:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def run(self, request: str, **kwargs: object) -> AgentResult:
        raise self.error


def _fake_live_class(shopper: ImmediateShopper | _ErrorShopper) -> type:
    class _FakeLive:
        last_kwargs: ClassVar[dict[str, object]] = {}

        def __init__(self, **kwargs: object) -> None:
            type(self).last_kwargs = kwargs

        async def __aenter__(self) -> ImmediateShopper | _ErrorShopper:
            return shopper

        async def __aexit__(self, *args: object) -> None:
            return None

    return _FakeLive


def test_streamlit_page_loads() -> None:
    at = _app_test()
    at.run()
    assert not at.exception
    assert at.title[0].value == "Mercari AI Shopper"
    assert "Offline fixtures" in at.checkbox[0].label
    assert at.button[0].label == "Search"


def test_streamlit_empty_request_shows_error() -> None:
    at = _app_test()
    at.run()
    at.button[0].click().run()
    assert not at.exception
    assert at.error[0].value == "Enter a shopping request."


def test_streamlit_search_shows_live_progress_and_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _fake_live_class(
        ImmediateShopper(
            _result(),
            progress_messages=('Searching Mercari… ("switch")', "Found 3 of 10 listings"),
        )
    )
    monkeypatch.setattr("mercari_shopper.app.LiveShopper", fake)
    at = _app_test()
    at.run()
    at.text_area[0].input("a used Nintendo Switch OLED under 30000 yen")
    at.button[0].click().run()
    assert not at.exception
    bodies = [str(item.value) for item in at.markdown]
    assert any("Searching Mercari" in text for text in bodies)
    assert any("Found 3 of 10 listings" in text for text in bodies)
    assert not any("How I got there" in str(item.label) for item in at.expander)


def test_streamlit_search_renders_top_three_cards(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _fake_live_class(ImmediateShopper(_result()))
    monkeypatch.setattr("mercari_shopper.app.LiveShopper", fake)
    at = _app_test()
    at.run()
    at.checkbox[0].check().run()
    at.text_area[0].input("a used Nintendo Switch OLED under 30000 yen")
    at.button[0].click().run()
    assert not at.exception
    assert fake.last_kwargs["allow_clarify"] is True
    assert fake.last_kwargs["source_kind"] == "fixture"
    assert any("Nintendo Switch 有機EL 本体" in str(item.value) for item in at.subheader)
    bodies = [str(item.value) for item in [*at.markdown, *at.text]]
    assert any("¥24,800" in text for text in bodies)
    assert any("jp.mercari.com/item/m10000000001" in text for text in bodies)
    assert any("sample-seller" in str(item.value) for item in at.caption)


def test_streamlit_question_shows_follow_up(monkeypatch: pytest.MonkeyPatch) -> None:
    question = AgentResult(
        summary="",
        recommendations=(),
        listings=(),
        steps=1,
        tool_calls=1,
        repairs=0,
        question="What kind of gift?",
    )
    fake = _fake_live_class(ImmediateShopper(question))
    monkeypatch.setattr("mercari_shopper.app.LiveShopper", fake)
    at = _app_test()
    at.run()
    at.text_area[0].input("a nice gift")
    at.button[0].click().run()
    assert not at.exception
    assert at.info[0].value == "What kind of gift?"
    assert any(button.label == "Continue" for button in at.button)
    assert at.session_state["pending_request"] == "a nice gift"


def test_streamlit_clarify_continue_runs_follow_up(monkeypatch: pytest.MonkeyPatch) -> None:
    question = AgentResult(
        summary="",
        recommendations=(),
        listings=(),
        steps=1,
        tool_calls=1,
        repairs=0,
        question="What kind of gift?",
    )
    shopper = ImmediateShopper(question)
    fake = _fake_live_class(shopper)
    monkeypatch.setattr("mercari_shopper.app.LiveShopper", fake)
    at = _app_test()
    at.run()
    at.text_area[0].input("a nice gift")
    at.button[0].click().run()

    shopper.result = _result()
    at.text_area[1].input("Nintendo Switch OLED")
    continue_btn = next(button for button in at.button if button.label == "Continue")
    continue_btn.click().run()
    assert not at.exception
    assert shopper.requests[1] == next_request("a nice gift", "Nintendo Switch OLED")
    assert any("Nintendo Switch 有機EL 本体" in str(item.value) for item in at.subheader)
    assert at.session_state["pending_question"] is None


def test_streamlit_agent_error_is_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _fake_live_class(_ErrorShopper(AgentBudgetError("out of steps")))
    monkeypatch.setattr("mercari_shopper.app.LiveShopper", fake)
    at = _app_test()
    at.run()
    at.text_area[0].input("switch")
    at.button[0].click().run()
    assert not at.exception
    assert "out of steps" in at.error[0].value


def test_streamlit_fair_value_and_risk_cons_on_cards(monkeypatch: pytest.MonkeyPatch) -> None:
    result = AgentResult(
        summary="Fair Switch.",
        recommendations=(
            Recommendation(item_id="m1", rank=1, reason="under typical", cons=["used"]),
        ),
        listings=(
            {
                "item_id": "m1",
                "name": "Switch",
                "price": 20000,
                "condition": "like new",
                "url": "https://jp.mercari.com/item/m1",
                "fair_value": {
                    "sample_size": 23,
                    "median": 24000,
                    "delta_pct": -16.7,
                    "confidence": "ok",
                    "label": "about 17% below typical sold price (based on 23 sold listings)",
                },
                "risk_flags": [{"code": "no_box", "label": "no box (箱なし)"}],
            },
        ),
        steps=2,
        tool_calls=2,
        repairs=0,
    )
    fake = _fake_live_class(ImmediateShopper(result))
    monkeypatch.setattr("mercari_shopper.app.LiveShopper", fake)
    at = _app_test()
    at.run()
    at.text_area[0].input("switch")
    at.button[0].click().run()
    assert not at.exception
    captions = [str(item.value) for item in at.caption]
    bodies = [str(item.value) for item in [*at.markdown, *at.text]]
    assert any("about 17% below typical sold price" in text for text in captions)
    assert any("no box" in text for text in bodies)
    assert any("used" in text for text in bodies)
