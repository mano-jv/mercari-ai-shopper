"""Streamlit UI wrapping LiveShopper. Secrets stay in the environment."""

from __future__ import annotations

import asyncio
from typing import Any

from mercari_shopper.agent.errors import AgentError
from mercari_shopper.agent.loop import AgentResult
from mercari_shopper.agent.prompts import PromptBuilder
from mercari_shopper.app import LiveShopper
from mercari_shopper.config import Settings
from mercari_shopper.llm.base import LLMClient
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
) -> AgentResult:
    async with LiveShopper(
        settings=settings,
        source_kind=source_kind(offline),
        provider=provider,
        model=model,
        llm=llm,
        allow_clarify=True,
    ) as shopper:
        return await shopper.run(request)


def run_search(
    request: str,
    *,
    offline: bool,
    provider: str | None = None,
    model: str | None = None,
    llm: LLMClient | None = None,
    settings: Settings | None = None,
) -> AgentResult:
    return asyncio.run(
        run_live(
            request,
            offline=offline,
            provider=provider,
            model=model,
            llm=llm,
            settings=settings,
        )
    )


def main() -> None:
    import streamlit as st

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
        st.markdown(
            "Live Mercari search works from a **home or office IP**. "
            "Cloud hosts (AWS, GCP, GitHub-hosted runners) often get **403**. "
            "Use Offline fixtures there. LLM keys still work in the cloud."
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
    try:
        with st.spinner("Searching Mercari…"):
            result = run_search(
                request,
                offline=offline,
                provider=provider,
                model=model,
            )
    except (AgentError, ValueError) as exc:
        st.session_state[_STATE_ERROR] = str(exc)
        st.session_state[_STATE_RESULT] = None
        st.session_state[_STATE_PENDING_REQUEST] = None
        st.session_state[_STATE_PENDING_QUESTION] = None
        return
    except Exception as exc:
        st.session_state[_STATE_ERROR] = f"{type(exc).__name__}: {exc}"
        st.session_state[_STATE_RESULT] = None
        st.session_state[_STATE_PENDING_REQUEST] = None
        st.session_state[_STATE_PENDING_QUESTION] = None
        return
    st.session_state[_STATE_RESULT] = result
    if needs_clarification(result):
        st.session_state[_STATE_PENDING_REQUEST] = original
        st.session_state[_STATE_PENDING_QUESTION] = result.question
    else:
        st.session_state[_STATE_PENDING_REQUEST] = None
        st.session_state[_STATE_PENDING_QUESTION] = None


def _render_result(st: Any, result: AgentResult) -> None:
    view = ResultViewBuilder().build(result)
    if view.summary:
        st.write(view.summary)
    if not view.items:
        return
    columns = st.columns(len(view.items))
    for column, item in zip(columns, view.items, strict=True):
        with column:
            _render_card(st, item)
    st.caption(f"{view.steps} steps · {view.tool_calls} tools · {view.elapsed_s:.1f}s")


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
