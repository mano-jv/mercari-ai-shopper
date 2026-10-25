"""One-off helper: render offline CLI output as SVG for docs/demo."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mercari_shopper.eval.cases import GOLDEN_CASES
from mercari_shopper.eval.harness import CatalogShopper
from mercari_shopper.presentation.render import PlainPresenter, RichPresenter, _rich_detail
from mercari_shopper.presentation.view import format_yen
from mercari_shopper.sources.factory import build_source

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "demo"


async def _agent_result() -> object:
    async with build_source("fixture") as source:
        shopper = CatalogShopper(source, GOLDEN_CASES)
        return await shopper.run(GOLDEN_CASES[0].query)


def _render_rich(result: object, path: Path, title_cmd: str) -> None:
    presenter = RichPresenter()
    view = presenter.view(result)
    console = Console(record=True, width=118)
    console.print(Text(title_cmd, style="bold cyan"))
    console.print()
    if view.summary:
        console.print(Panel(Text(view.summary), title="Mercari shopper", expand=False))
    if view.items:
        table = Table(show_header=True, header_style="bold", expand=True)
        table.add_column("#", style="bold", width=3)
        table.add_column("Item")
        table.add_column("Price", justify="right")
        table.add_column("Condition")
        for item in view.items:
            name = Text(item.name)
            table.add_row(str(item.rank), name, format_yen(item.price), Text(item.condition))
        console.print(table)
        for item in view.items:
            console.print(_rich_detail(item))
    console.save_svg(str(path))


def _run_module(*args: str) -> str:
    completed = subprocess.run(
        [sys.executable, "-m", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=ROOT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout)
    return completed.stdout.rstrip()


def _svg_to_png(svg_path: Path) -> None:
    from playwright.sync_api import sync_playwright

    png_path = svg_path.with_suffix(".png")
    svg_markup = svg_path.read_text(encoding="utf-8")
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<style>html,body{margin:0;background:#0c0c0c;}</style></head>"
        f"<body>{svg_markup}</body></html>"
    )
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 2000})
        page.set_content(html, wait_until="domcontentloaded")
        page.screenshot(path=str(png_path), full_page=True, timeout=120_000)
        browser.close()


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result = await _agent_result()
    _render_rich(
        result,
        OUT / "cli-recommendations.svg",
        'python -m mercari_shopper --offline -q "a used Nintendo Switch OLED under 30000 yen in good condition"',
    )

    plain = PlainPresenter(verbose=True).render(result)
    console = Console(record=True, width=118)
    console.print(
        Text(
            'python -m mercari_shopper --plain --verbose --offline -q "..."',
            style="bold cyan",
        )
    )
    console.print()
    console.print(plain.rstrip())
    console.save_svg(str(OUT / "cli-plain-verbose.svg"))

    sources_out = _run_module(
        "mercari_shopper.sources",
        "--q",
        "switch",
        "--source",
        "fixture",
        "--max",
        "5",
    )
    c_sources = Console(record=True, width=118)
    c_sources.print(
        Text(
            "python -m mercari_shopper.sources --q switch --source fixture --max 5",
            style="bold cyan",
        )
    )
    c_sources.print()
    c_sources.print(sources_out)
    c_sources.save_svg(str(OUT / "cli-sources-search.svg"))

    eval_out = _run_module("mercari_shopper", "--eval")
    c_eval = Console(record=True, width=118)
    c_eval.print(Text("python -m mercari_shopper --eval", style="bold cyan"))
    c_eval.print()
    c_eval.print(eval_out)
    c_eval.save_svg(str(OUT / "cli-eval.svg"))


def _png_exports() -> None:
    for svg in sorted(OUT.glob("cli-*.svg")):
        _svg_to_png(svg)


if __name__ == "__main__":
    asyncio.run(main())
    _png_exports()
