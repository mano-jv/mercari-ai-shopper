"""Command-line entry: one-shot query, interactive REPL, or golden-query eval."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from typing import TextIO

from mercari_shopper import __version__
from mercari_shopper.agent.errors import AgentError
from mercari_shopper.agent.loop import Shopper
from mercari_shopper.app import LiveShopper
from mercari_shopper.config import load_settings
from mercari_shopper.eval.harness import CatalogShopper, EvalHarness
from mercari_shopper.logging import configure_logging
from mercari_shopper.presentation.interactive import InteractiveSession, Stdio
from mercari_shopper.presentation.render import Presenter, build_presenter
from mercari_shopper.sources.factory import build_source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mercari-shopper",
        description="Natural-language shopping agent for Mercari Japan.",
    )
    parser.add_argument(
        "request",
        nargs="?",
        help="Shopping request in Japanese or English.",
    )
    parser.add_argument(
        "-q",
        "--query",
        dest="query",
        help="Shopping request (same as the positional argument).",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Prompt for requests until quit.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        dest="run_eval",
        help="Run the golden-query harness against offline fixtures.",
    )
    parser.add_argument(
        "--provider",
        choices=("anthropic", "openai", "vertex"),
        default=None,
        help="Override LLM_PROVIDER.",
    )
    parser.add_argument("--model", default=None, help="Override LLM_MODEL.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use recorded Mercari fixtures instead of the live site.",
    )
    parser.add_argument("--json", action="store_true", help="Print the result as JSON.")
    parser.add_argument("--plain", action="store_true", help="Disable Rich formatting.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log tool calls and print per-step timings.",
    )
    parser.add_argument("--version", action="store_true", help="Print the version and exit.")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    shopper: Shopper | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    _configure_stdio()
    in_stream = stdin if stdin is not None else sys.stdin
    out_stream = stdout if stdout is not None else sys.stdout
    err_stream = stderr if stderr is not None else sys.stderr
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.version:
        out_stream.write(f"mercari-ai-shopper {__version__}\n")
        return 0
    request = _request_text(args)
    if args.run_eval and (args.interactive or request):
        err_stream.write("error: --eval cannot be combined with a query or --interactive\n")
        return 2
    if args.interactive and request:
        err_stream.write("error: --interactive cannot be combined with a query\n")
        return 2
    if not args.run_eval and not args.interactive and not request:
        parser.print_help(err_stream)
        return 2
    configure_logging(verbose=args.verbose)
    presenter = build_presenter(
        json_output=args.json,
        plain=args.plain,
        verbose=args.verbose,
        tty=out_stream.isatty(),
    )
    try:
        return asyncio.run(
            _dispatch(
                args,
                request=request,
                presenter=presenter,
                shopper=shopper,
                stdin=in_stream,
                stdout=out_stream,
                stderr=err_stream,
            )
        )
    except ValueError as exc:
        err_stream.write(f"error: {exc}\n")
        return 1


async def _dispatch(
    args: argparse.Namespace,
    *,
    request: str,
    presenter: Presenter,
    shopper: Shopper | None,
    stdin: TextIO,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    if args.run_eval:
        return await _run_eval(shopper, stdout)
    if args.interactive:
        return await _run_interactive(
            args, presenter=presenter, shopper=shopper, stdin=stdin, stdout=stdout, stderr=stderr
        )
    return await _run_once(
        args, request=request, presenter=presenter, shopper=shopper, stdout=stdout, stderr=stderr
    )


async def _run_once(
    args: argparse.Namespace,
    *,
    request: str,
    presenter: Presenter,
    shopper: Shopper | None,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    if shopper is not None:
        return await _shop(shopper, request, presenter, stdout, stderr)
    try:
        settings = load_settings()
    except ValueError as exc:
        stderr.write(f"error: {exc}\n")
        return 1
    source_kind = "fixture" if args.offline else None
    try:
        async with LiveShopper(
            settings=settings,
            source_kind=source_kind,
            provider=args.provider,
            model=args.model,
        ) as live:
            return await _shop(live, request, presenter, stdout, stderr)
    except ValueError as exc:
        stderr.write(f"error: {exc}\n")
        return 1


async def _run_interactive(
    args: argparse.Namespace,
    *,
    presenter: Presenter,
    shopper: Shopper | None,
    stdin: TextIO,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    io = Stdio(stdin, stdout, stderr)
    if shopper is not None:
        return await InteractiveSession(shopper, presenter, io).run()
    try:
        settings = load_settings()
    except ValueError as exc:
        stderr.write(f"error: {exc}\n")
        return 1
    source_kind = "fixture" if args.offline else None
    try:
        async with LiveShopper(
            settings=settings,
            source_kind=source_kind,
            provider=args.provider,
            model=args.model,
        ) as live:
            return await InteractiveSession(live, presenter, io).run()
    except ValueError as exc:
        stderr.write(f"error: {exc}\n")
        return 1


async def _run_eval(shopper: Shopper | None, stdout: TextIO) -> int:
    harness = EvalHarness()
    if shopper is not None:
        report = await harness.run(shopper)
    else:
        async with build_source("fixture") as source:
            report = await harness.run(CatalogShopper(source, harness.cases))
    stdout.write(report.render())
    return 0 if report.ok else 1


async def _shop(
    shopper: Shopper,
    request: str,
    presenter: Presenter,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    try:
        result = await shopper.run(request)
    except AgentError as exc:
        presenter.emit_error(exc, stderr)
        return 1
    presenter.emit(result, stdout)
    return 0


def _request_text(args: argparse.Namespace) -> str:
    flagged = (args.query or "").strip()
    positional = (args.request or "").strip()
    return flagged or positional


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    raise SystemExit(main())
