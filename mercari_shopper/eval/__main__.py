"""Run the golden-query harness: python -m mercari_shopper.eval [--offline]."""

from __future__ import annotations

import sys

from mercari_shopper.cli import main

if __name__ == "__main__":
    argv = ["--eval", *sys.argv[1:]]
    raise SystemExit(main(argv))
