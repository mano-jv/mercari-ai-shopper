"""Interactive REPL. Template: read → dispatch → render, until quit."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TextIO

from mercari_shopper.agent.loop import Shopper
from mercari_shopper.agent.prompts import PromptBuilder
from mercari_shopper.presentation.render import Presenter

_QUIT = frozenset({"q", "quit", "exit", ":q"})
_HELP = frozenset({"help", "?"})

_BANNER = "Mercari Japan shopper. Type a request, or quit to exit."
_HELP_TEXT = "Commands: a shopping request, help, quit."


class UserIO(Protocol):
    """Port so tests can drive the session without a real terminal."""

    @property
    def out(self) -> TextIO:
        """Stream presenters emit recommendations to."""

    @property
    def err(self) -> TextIO:
        """Stream presenters emit failures to."""

    def read(self, prompt: str) -> str | None:
        """Return one line, or None on EOF."""

    def write(self, text: str) -> None:
        """Write a prompt or banner to the user-facing output stream."""

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

    @property
    def out(self) -> TextIO:
        return self._stdout

    @property
    def err(self) -> TextIO:
        return self._stderr

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
        self._stdout.flush()

    def write_error(self, text: str) -> None:
        self._stderr.write(text if text.endswith("\n") else text + "\n")
        self._stderr.flush()


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
            except Exception as exc:
                self._presenter.emit_error(exc, self._io.err)
                continue
            if result.question and not result.recommendations:
                self._presenter.emit(result, self._io.out)
                answer = self._io.read("clarify> ")
                if answer is None:
                    return 0
                follow = answer.strip()
                if not follow:
                    continue
                if follow.casefold() in _QUIT:
                    return 0
                if follow.casefold() in _HELP:
                    self._io.write(_HELP_TEXT)
                    continue
                try:
                    result = await self._shopper.run(
                        PromptBuilder().with_clarification(text, follow)
                    )
                except Exception as exc:
                    self._presenter.emit_error(exc, self._io.err)
                    continue
            self._presenter.emit(result, self._io.out)
