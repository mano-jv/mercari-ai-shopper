"""Structured logs as one JSON object per line."""

from __future__ import annotations

import json
import logging
import sys

_LOGGER = logging.getLogger("mercari_shopper")


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stderr, force=True)
    for name in ("httpx", "httpcore", "playwright", "google", "urllib3", "anthropic"):
        logging.getLogger(name).setLevel(logging.WARNING)


def log_event(event: str, **fields: object) -> None:
    _LOGGER.info(json.dumps({"event": event, **fields}, default=str, ensure_ascii=False))
