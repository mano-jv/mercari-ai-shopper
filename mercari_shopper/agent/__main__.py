"""Run one shopping request: python -m mercari_shopper.agent --q TEXT."""

from __future__ import annotations

import sys

from mercari_shopper.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
