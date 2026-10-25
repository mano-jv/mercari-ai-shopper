"""Streamlit UI wrapping LiveShopper. Secrets stay in the environment."""

from __future__ import annotations

import asyncio
import os
import queue
import threading
from collections.abc import Iterator
from typing import Any

from mercari_shopper.agent.errors import AgentError
from mercari_shopper.agent.loop import AgentResult
from mercari_shopper.agent.progress import (
    STATUS_START,
    ProgressCallback,
    ProgressStep,
    ProgressTimeline,
)
from mercari_shopper.agent.prompts import PromptBuilder
from mercari_shopper.app import LiveShopper
from mercari_shopper.config import Settings
from mercari_shopper.llm.base import LLMClient
from mercari_shopper.logging import configure_logging
from mercari_shopper.presentation.view import (
    RecommendationView,
    ResultViewBuilder,
    SellerView,
    format_yen,
)

ENV_DEFAULT = "(env default)"
PROVIDERS = (ENV_DEFAULT, "anthropic", "openai", "vertex")

_STATE_RESULT = "result"
_STATE_ERROR = "error"
_STATE_PENDING_REQUEST = "pending_request"
_STATE_PENDING_QUESTION = "pending_question"


def source_kind(offline: bool) -> str | None:
    """Fixture source when offline; otherwise LiveShopper uses MERCARI_SOURCE."""
    return "fixture" if offline else None


def optional_override(value: str | None) -> str | None:
    text = (value or "").strip()
    if not text or text == ENV_DEFAULT:
        return None
    return text


def needs_clarification(result: AgentResult) -> bool:
    return bool(result.question) and not result.recommendations


def next_request(original: str, answer: str) -> str:
    """Second pass after ask_user. Matches the CLI follow-up."""
    return PromptBuilder().with_clarification(original, answer)


def seller_line(seller: SellerView | None) -> str:
    if seller is None:
        return ""
    parts: list[str] = []
    if seller.name:
        parts.append(seller.name)
    if seller.star_rating is not None:
        parts.append(f"★{seller.star_rating:g}")
    if seller.num_ratings is not None:
        parts.append(f"({seller.num_ratings})")
    return " ".join(parts)


async def run_live(
    request: str,
    *,
    offline: bool,
    provider: str | None = None,
    model: str | None = None,
    llm: LLMClient | None = None,
    settings: Settings | None = None,
    on_progress: ProgressCallback | None = None,
) -> AgentResult:
    async with LiveShopper(
        settings=settings,
        source_kind=source_kind(offline),
        provider=provider,
        model=model,
        llm=llm,
        allow_clarify=True,
    ) as shopper:
        return await shopper.run(request, on_progress=on_progress)


class ShopperHolder:
    """One event loop and one LiveShopper per (offline, provider, model).

    Streamlit reruns the script on every search and `asyncio.run` would close
    the loop, so Anthropic/OpenAI would redo TLS and Vertex would redo token
    setup. Keep the clients on a background loop instead. Progress is queued
    back to the Streamlit thread.
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="mercari-shopper", daemon=True
        )
        self._thread.start()
        self._sessions: dict[tuple[object, ...], LiveShopper] = {}

    def search(
        self,
        request: str,
        *,
        offline: bool,
        provider: str | None = None,
        model: str | None = None,
        settings: Settings | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> AgentResult:
        incoming: queue.Queue[str] = queue.Queue()

        def bridge(message: str) -> None:
            incoming.put(message)

        future = asyncio.run_coroutine_threadsafe(
            self._search(
                request,
                offline=offline,
                provider=provider,
                model=model,
                settings=settings,
                on_progress=bridge if on_progress is not None else None,
            ),
            self._loop,
        )
        while not future.done():
            _drain_progress(incoming, on_progress, block=True)
        _drain_progress(incoming, on_progress, block=False)
        return future.result()

    async def _search(
        self,
        request: str,
        *,
        offline: bool,
        provider: str | None,
        model: str | None,
        settings: Settings | None,
        on_progress: ProgressCallback | None,
    ) -> AgentResult:
        key = (offline, provider, model)
        shopper = self._sessions.get(key)
        if shopper is None:
            created = LiveShopper(
                settings=settings,
                source_kind=source_kind(offline),
                provider=provider,
                model=model,
                allow_clarify=True,
            )
            await created.__aenter__()
            shopper = self._sessions.setdefault(key, created)
            if shopper is not created:
                await created.__aexit__(None, None, None)
        return await shopper.run(request, on_progress=on_progress)

    def close(self) -> None:
        async def shutdown() -> None:
            for shopper in self._sessions.values():
                await shopper.__aexit__(None, None, None)
            self._sessions.clear()

        asyncio.run_coroutine_threadsafe(shutdown(), self._loop).result(timeout=10)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)


def _drain_progress(
    incoming: queue.Queue[str],
    on_progress: ProgressCallback | None,
    *,
    block: bool,
) -> None:
    while True:
        try:
            message = incoming.get(timeout=0.05) if block else incoming.get_nowait()
        except queue.Empty:
            return
        if on_progress is not None:
            on_progress(message)
        block = False


def _new_shopper_holder() -> ShopperHolder:
    return ShopperHolder()


def _under_pytest() -> bool:
    return bool(os.environ.get("PYTEST_CURRENT_TEST"))


def run_search(
    request: str,
    *,
    offline: bool,
    provider: str | None = None,
    model: str | None = None,
    llm: LLMClient | None = None,
    settings: Settings | None = None,
    on_progress: ProgressCallback | None = None,
) -> AgentResult:
    if llm is not None or _under_pytest():
        return asyncio.run(
            run_live(
                request,
                offline=offline,
                provider=provider,
                model=model,
                llm=llm,
                settings=settings,
                on_progress=on_progress,
            )
        )
    import streamlit as st

    holder = st.cache_resource(_new_shopper_holder)()
    return holder.search(
        request,
        offline=offline,
        provider=provider,
        model=model,
        settings=settings,
        on_progress=on_progress,
    )


def summary_stream_chunks(text: str) -> Iterator[str]:
    """Character chunks for ``st.write_stream`` on the submit summary."""
    yield from text


def main() -> None:
    import streamlit as st

    configure_logging()
    st.set_page_config(page_title="Mercari AI Shopper", layout="wide")
    _init_state(st)

    st.title("Mercari AI Shopper")
    st.caption(
        "Natural-language shopping for Mercari Japan. "
        "API keys stay in the environment and are never sent to the browser."
    )

    with st.sidebar:
        offline = st.checkbox(
            "Offline fixtures",
            help="Use recorded Mercari listings instead of the live site.",
        )
        provider_label = st.selectbox("LLM provider", PROVIDERS)
        model = st.text_input(
            "Model override",
            placeholder="Leave blank for LLM_MODEL / the provider default",
        )

    provider = optional_override(str(provider_label) if provider_label else None)
    model_override = optional_override(model)

    with st.form("search"):
        request = st.text_area(
            "Shopping request (English or Japanese)",
            placeholder="a used Nintendo Switch OLED under 30000 yen in good condition",
            height=100,
        )
        submitted = st.form_submit_button("Search", type="primary")

    if submitted:
        _handle_search(
            st,
            request,
            offline=offline,
            provider=provider,
            model=model_override,
        )

    pending_question = st.session_state.get(_STATE_PENDING_QUESTION)
    if pending_question:
        st.info(pending_question)
        with st.form("clarify"):
            answer = st.text_area("Your answer", height=80)
            continued = st.form_submit_button("Continue")
        if continued:
            _handle_clarify(
                st,
                answer,
                offline=offline,
                provider=provider,
                model=model_override,
            )

    error = st.session_state.get(_STATE_ERROR)
    if error:
        st.error(error)

    result = st.session_state.get(_STATE_RESULT)
    if isinstance(result, AgentResult) and not needs_clarification(result):
        _render_result(st, result)


def _init_state(st: Any) -> None:
    for key, default in (
        (_STATE_RESULT, None),
        (_STATE_ERROR, None),
        (_STATE_PENDING_REQUEST, None),
        (_STATE_PENDING_QUESTION, None),
    ):
        if key not in st.session_state:
            st.session_state[key] = default


def _handle_search(
    st: Any,
    request: str,
    *,
    offline: bool,
    provider: str | None,
    model: str | None,
) -> None:
    st.session_state[_STATE_ERROR] = None
    st.session_state[_STATE_RESULT] = None
    st.session_state[_STATE_PENDING_REQUEST] = None
    st.session_state[_STATE_PENDING_QUESTION] = None
    text = request.strip()
    if not text:
        st.session_state[_STATE_ERROR] = "Enter a shopping request."
        return
    _store_run(st, text, original=text, offline=offline, provider=provider, model=model)


def _handle_clarify(
    st: Any,
    answer: str,
    *,
    offline: bool,
    provider: str | None,
    model: str | None,
) -> None:
    original = str(st.session_state.get(_STATE_PENDING_REQUEST) or "")
    text = answer.strip()
    if not text:
        st.session_state[_STATE_ERROR] = "Enter an answer to continue."
        return
    st.session_state[_STATE_ERROR] = None
    _store_run(
        st,
        next_request(original, text),
        original=original,
        offline=offline,
        provider=provider,
        model=model,
    )


def _store_run(
    st: Any,
    request: str,
    *,
    original: str,
    offline: bool,
    provider: str | None,
    model: str | None,
) -> None:
    with st.status(STATUS_START, expanded=True) as status:
        timeline = ProgressTimeline()
        log = st.empty()

        def on_progress(message: str) -> None:
            status.update(label=message)
            log.markdown(_render_timeline(timeline.advance(message)))

        try:
            result = run_search(
                request,
                offline=offline,
                provider=provider,
                model=model,
                on_progress=on_progress,
            )
        except (AgentError, ValueError) as exc:
            log.markdown(_render_timeline(timeline.finish()))
            status.update(label="Something went wrong", state="error")
            _clear_result(st, str(exc))
            return
        except Exception as exc:
            log.markdown(_render_timeline(timeline.finish()))
            status.update(label="Something went wrong", state="error")
            _clear_result(st, f"{type(exc).__name__}: {exc}")
            return
        log.markdown(_render_timeline(timeline.finish()))
        status.update(label="Done", state="complete", expanded=False)
    st.session_state[_STATE_RESULT] = result
    if needs_clarification(result):
        st.session_state[_STATE_PENDING_REQUEST] = original
        st.session_state[_STATE_PENDING_QUESTION] = result.question
    else:
        st.session_state[_STATE_PENDING_REQUEST] = None
        st.session_state[_STATE_PENDING_QUESTION] = None


def _clear_result(st: Any, error: str) -> None:
    st.session_state[_STATE_ERROR] = error
    st.session_state[_STATE_RESULT] = None
    st.session_state[_STATE_PENDING_REQUEST] = None
    st.session_state[_STATE_PENDING_QUESTION] = None


def _render_timeline(steps: tuple[ProgressStep, ...]) -> str:
    if not steps:
        return ""
    lines: list[str] = []
    for step in steps:
        mark = "✅" if step.done else "⏳"
        lines.append(f"- {mark} {step.text}")
    return "\n".join(lines)


def _render_result(st: Any, result: AgentResult) -> None:
    view = ResultViewBuilder().build(result)
    if view.summary:
        st.write_stream(summary_stream_chunks(view.summary))
    if view.items:
        columns = st.columns(len(view.items))
        for column, item in zip(columns, view.items, strict=True):
            with column:
                _render_card(st, item)


def _render_card(st: Any, item: RecommendationView) -> None:
    st.subheader(f"{item.rank}. {item.name}")
    st.write(f"**{format_yen(item.price)}** · {item.condition}")
    if item.fair_value is not None:
        st.caption(item.fair_value.label)
    st.write(item.reason)
    for con in item.cons:
        st.markdown(f"- {con}")
    seller = seller_line(item.seller)
    if seller:
        st.caption(seller)
    if item.url:
        st.markdown(f"[View on Mercari]({item.url})")


if __name__ == "__main__":
    main()
