"""Run one shopping request: python -m mercari_shopper.agent --q TEXT."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from mercari_shopper.agent.errors import AgentError
from mercari_shopper.agent.factory import build_agent
from mercari_shopper.agent.loop import AgentResult
from mercari_shopper.config import load_settings
from mercari_shopper.llm.factory import build_llm
from mercari_shopper.logging import configure_logging
from mercari_shopper.sources.factory import build_source
from mercari_shopper.tools.registry import ToolRegistry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mercari_shopper.agent",
        description="Run the Phase 3 shopping loop for one natural-language request.",
    )
    parser.add_argument("--q", required=True, help="Shopping request in Japanese or English.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use recorded Mercari fixtures instead of the live site.",
    )
    parser.add_argument(
        "--provider",
        choices=("anthropic", "openai", "vertex"),
        default=None,
        help="Override LLM_PROVIDER.",
    )
    parser.add_argument("--model", default=None, help="Override LLM_MODEL.")
    parser.add_argument("--json", action="store_true", help="Print the result as JSON.")
    parser.add_argument("--verbose", action="store_true")
    return parser


async def run(args: argparse.Namespace) -> int:
    configure_logging(verbose=args.verbose)
    try:
        settings = load_settings()
        source_kind = "fixture" if args.offline else None
        llm = build_llm(args.provider, settings=settings, model=args.model)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    async with build_source(source_kind, settings=settings) as source:
        try:
            agent = build_agent(llm, ToolRegistry.for_source(source), settings=settings)
            result = await agent.run(args.q)
        except AgentError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        finally:
            await llm.aclose()
    if args.json:
        print(json.dumps(_result_payload(result), ensure_ascii=False, indent=2))
        return 0
    _print_result(result)
    return 0


def _print_result(result: AgentResult) -> None:
    print(result.summary)
    for rec, listing in zip(result.recommendations, result.listings, strict=True):
        name = listing.get("name", rec.item_id)
        price = listing.get("price", "?")
        print(f"{rec.rank}. {name} ¥{price} ({rec.item_id})")
        print(f"   {rec.reason}")


def _result_payload(result: AgentResult) -> dict[str, object]:
    return {
        "summary": result.summary,
        "recommendations": [rec.model_dump() for rec in result.recommendations],
        "listings": list(result.listings),
        "steps": result.steps,
        "tool_calls": result.tool_calls,
        "repairs": result.repairs,
    }


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    args = build_parser().parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
