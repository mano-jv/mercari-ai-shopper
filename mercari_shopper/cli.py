"""Command-line entry point.

Phase 0 only prints the version. Search and the agent loop arrive in later phases.
"""

from __future__ import annotations

import argparse

from mercari_shopper import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mercari-shopper",
        description="Natural-language shopping agent for Mercari Japan.",
    )
    parser.add_argument("--version", action="store_true", help="Print the version and exit.")
    return parser


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    print(f"mercari-ai-shopper {__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
