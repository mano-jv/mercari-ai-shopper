"""Search Mercari without the agent loop: python -m mercari_shopper.sources --q KEYWORD."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from mercari_shopper.domain.models import SearchQuery, condition_label
from mercari_shopper.logging import configure_logging
from mercari_shopper.sources.factory import build_source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mercari_shopper.sources",
        description="Run a Mercari search through the Phase 1 data layer.",
    )
    parser.add_argument("--q", required=True, help="Search keyword.")
    parser.add_argument("--max", type=int, default=5, help="Page size (1-120).")
    parser.add_argument("--price-min", type=int, default=None)
    parser.add_argument("--price-max", type=int, default=None)
    parser.add_argument(
        "--source",
        choices=("auto", "dpop", "browser", "fixture"),
        default=None,
        help="Override MERCARI_SOURCE. Default is auto (or fixture if set in the env).",
    )
    parser.add_argument("--json", action="store_true", help="Print listings as JSON.")
    parser.add_argument("--verbose", action="store_true")
    return parser


async def run(args: argparse.Namespace) -> int:
    configure_logging(verbose=args.verbose)
    query = SearchQuery(
        keyword=args.q,
        price_min=args.price_min,
        price_max=args.price_max,
        page_size=args.max,
    )
    async with build_source(args.source) as source:
        result = await source.search(query)
    if args.json:
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return 0
    print(f"{result.num_found} found, showing {len(result.items)}")
    for listing in result.items:
        print(
            f"{listing.item_id}\t¥{listing.price}\t{condition_label(listing.condition_id)}"
            f"\t{listing.name}\t{listing.url}"
        )
    return 0


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
