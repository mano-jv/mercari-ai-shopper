"""Interactive REPL. Template: read → dispatch → render, until quit."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TextIO

from mercari_shopper.agent.errors import AgentError
from mercari_shopper.agent.loop import Shopper
from mercari_shopper.presentation.render import Presenter

_QUIT = frozenset({"q", "quit", "exit", ":q"})
_HELP = frozenset({"help", "?"})

_BANNER = "Mercari Japan shopper. Type a request, or quit to exit."
_HELP_TEXT = "Commands: a shopping request, help, quit."


class UserIO(Protocol):
    """Port so tests can drive the session without a real terminal."""

    def read(self, prompt: str) -> str | None:
        """Return one line, or None on EOF."""

    def write(self, text: str) -> None:
        """Write to the user-facing output stream."""

    def write_error(self, text: str) -> None:
        """Write a failure. Defaults can reuse `write`."""


class Stdio:
    def __init__(
        self,
        stdin: TextIO,
        stdout: TextIO,
        stderr: TextIO | None = None,
        *,
        input_fn: Callable[[str], str] | None = None,
    ) -> None:
        self._stdin = stdin
        self._stdout = stdout
        self._stderr = stderr or stdout
        self._input_fn = input_fn

    def read(self, prompt: str) -> str | None:
        getter = self._input_fn
        if getter is not None:
            try:
                return getter(prompt)
            except EOFError:
                return None
        self._stdout.write(prompt)
        self._stdout.flush()
        line = self._stdin.readline()
        if line == "":
            return None
        return line

    def write(self, text: str) -> None:
        self._stdout.write(text if text.endswith("\n") else text + "\n")

    def write_error(self, text: str) -> None:
        self._stderr.write(text if text.endswith("\n") else text + "\n")


class InteractiveSession:
    """Loop over user lines. Empty input is ignored; quit words end the session."""

    prompt = "shop> "

    def __init__(self, shopper: Shopper, presenter: Presenter, io: UserIO) -> None:
        self._shopper = shopper
        self._presenter = presenter
        self._io = io

    async def run(self) -> int:
        self._io.write(_BANNER)
        while True:
            raw = self._io.read(self.prompt)
            if raw is None:
                return 0
            text = raw.strip()
            if not text:
                continue
            folded = text.casefold()
            if folded in _QUIT:
                return 0
            if folded in _HELP:
                self._io.write(_HELP_TEXT)
                continue
            try:
                result = await self._shopper.run(text)
            except AgentError as exc:
                self._io.write_error(self._presenter.render_error(exc))
                continue
            self._io.write(self._presenter.render(result))
