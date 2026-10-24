"""Presenter strategies: plain text, Rich, and JSON."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, TextIO

from mercari_shopper.agent.loop import AgentResult
from mercari_shopper.presentation.view import (
    RecommendationView,
    ResultView,
    ResultViewBuilder,
    SellerView,
    format_yen,
)


class Presenter(ABC):
    """Strategy: turn an AgentResult into text the CLI can print."""

    def __init__(self, *, verbose: bool = False, builder: ResultViewBuilder | None = None) -> None:
        self._verbose = verbose
        self._builder = builder or ResultViewBuilder()

    @abstractmethod
    def render(self, result: AgentResult) -> str:
        """Stable text. Used by tests and by `emit` unless a subclass overrides it."""

    def emit(self, result: AgentResult, stream: TextIO) -> None:
        text = self.render(result)
        stream.write(text if text.endswith("\n") else text + "\n")
        _flush(stream)

    def render_error(self, error: BaseException) -> str:
        return f"error: {error}"

    def emit_error(self, error: BaseException, stream: TextIO) -> None:
        stream.write(self.render_error(error) + "\n")
        _flush(stream)

    def view(self, result: AgentResult) -> ResultView:
        return self._builder.build(result)


class PlainPresenter(Presenter):
    """Readable top-3 for pipes, logs, and snapshot tests."""

    def render(self, result: AgentResult) -> str:
        view = self.view(result)
        lines: list[str] = []
        if view.summary:
            lines.append(view.summary)
            lines.append("")
        for item in view.items:
            lines.extend(_plain_card(item))
            lines.append("")
        if self._verbose:
            lines.extend(_plain_trace(view))
        text = "\n".join(lines).rstrip()
        return text + "\n" if text else ""


class JsonPresenter(Presenter):
    """Machine-readable result. Verbose adds the per-step trace."""

    def render(self, result: AgentResult) -> str:
        view = self.view(result)
        payload: dict[str, Any] = {
            "summary": view.summary,
            "recommendations": [_json_card(item) for item in view.items],
            "steps": view.steps,
            "tool_calls": view.tool_calls,
            "repairs": view.repairs,
            "elapsed_s": view.elapsed_s,
        }
        if self._verbose:
            payload["trace"] = [
                {"step": step.step, "elapsed_s": step.elapsed_s, "tools": list(step.tool_names)}
                for step in view.trace
            ]
        return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"

    def render_error(self, error: BaseException) -> str:
        return (
            json.dumps({"error": str(error), "type": type(error).__name__}, ensure_ascii=False)
            + "\n"
        )


class RichPresenter(Presenter):
    """Colourful tables when stdout is a terminal. `render` stays snapshot-friendly."""

    def render(self, result: AgentResult) -> str:
        return PlainPresenter(verbose=self._verbose, builder=self._builder).render(result)

    def emit(self, result: AgentResult, stream: TextIO) -> None:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        view = self.view(result)
        console = Console(file=stream, highlight=False)
        if view.summary:
            console.print(Panel(Text(view.summary), title="Mercari shopper", expand=False))
        table = Table(show_header=True, header_style="bold", expand=True)
        table.add_column("#", style="bold", width=3)
        table.add_column("Item")
        table.add_column("Price", justify="right")
        table.add_column("Condition")
        for item in view.items:
            name = Text(item.name)
            if item.is_shop_item:
                name.append("  [shop]", style="dim")
            if item.is_auction:
                name.append("  [auction]", style="dim")
            table.add_row(str(item.rank), name, format_yen(item.price), Text(item.condition))
        console.print(table)
        for item in view.items:
            console.print(_rich_detail(item))
        if self._verbose:
            for line in _plain_trace(view):
                console.print(Text(line, style="dim"))
        _flush(stream)


class PresenterFactory:
    """Picks a presenter from CLI flags. `--json` wins over `--plain`."""

    def build(
        self,
        *,
        json_output: bool = False,
        plain: bool = False,
        verbose: bool = False,
        tty: bool = False,
    ) -> Presenter:
        if json_output:
            return JsonPresenter(verbose=verbose)
        if plain or not tty:
            return PlainPresenter(verbose=verbose)
        return RichPresenter(verbose=verbose)


def build_presenter(
    *,
    json_output: bool = False,
    plain: bool = False,
    verbose: bool = False,
    tty: bool = False,
) -> Presenter:
    return PresenterFactory().build(json_output=json_output, plain=plain, verbose=verbose, tty=tty)


def _plain_card(item: RecommendationView) -> list[str]:
    tags: list[str] = []
    if item.is_shop_item:
        tags.append("shop")
    if item.is_auction:
        tags.append("auction")
    tag_text = f"  ({', '.join(tags)})" if tags else ""
    lines = [
        f"{item.rank}. {item.name}{tag_text}",
        f"   {format_yen(item.price)}  ·  {item.condition}",
    ]
    if item.url:
        lines.append(f"   {item.url}")
    lines.append(f"   {item.reason}")
    for pro in item.pros:
        lines.append(f"   + {pro}")
    for con in item.cons:
        lines.append(f"   - {con}")
    seller_line = _seller_line(item.seller)
    if seller_line:
        lines.append(f"   {seller_line}")
    return lines


def _plain_trace(view: ResultView) -> list[str]:
    lines = [
        "---",
        (
            f"steps: {view.steps}  tools: {view.tool_calls}  "
            f"repairs: {view.repairs}  {view.elapsed_s:.2f}s"
        ),
    ]
    for step in view.trace:
        tools = ", ".join(step.tool_names) if step.tool_names else "(no tool call)"
        lines.append(f"{step.step}. {tools} ({step.elapsed_s:.2f}s)")
    return lines


def _json_card(item: RecommendationView) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "rank": item.rank,
        "item_id": item.item_id,
        "name": item.name,
        "price": item.price,
        "currency": "JPY",
        "condition": item.condition,
        "url": item.url,
        "reason": item.reason,
        "pros": list(item.pros),
        "cons": list(item.cons),
        "is_shop_item": item.is_shop_item,
        "is_auction": item.is_auction,
        "seller": None,
    }
    if item.seller is not None:
        payload["seller"] = {
            "name": item.seller.name,
            "star_rating": item.seller.star_rating,
            "num_ratings": item.seller.num_ratings,
        }
    return payload


def _seller_line(seller: SellerView | None) -> str:
    if seller is None:
        return ""
    parts = ["seller:"]
    if seller.name:
        parts.append(seller.name)
    if seller.star_rating is not None:
        parts.append(f"★{seller.star_rating:g}")
    if seller.num_ratings is not None:
        parts.append(f"({seller.num_ratings})")
    return " ".join(parts) if len(parts) > 1 else ""


def _rich_detail(item: RecommendationView) -> object:
    from rich.text import Text

    text = Text()
    text.append(f"{item.rank}. ", style="bold")
    text.append(item.reason)
    if item.url:
        text.append("\n")
        text.append(item.url)
    for pro in item.pros:
        text.append("\n")
        text.append(f"+ {pro}")
    for con in item.cons:
        text.append("\n")
        text.append(f"- {con}")
    seller_line = _seller_line(item.seller)
    if seller_line:
        text.append("\n")
        text.append(seller_line)
    return text


def _flush(stream: TextIO) -> None:
    flush = getattr(stream, "flush", None)
    if flush is not None:
        flush()
