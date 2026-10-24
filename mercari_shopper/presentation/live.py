"""Live activity timeline on stderr while the agent runs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TextIO

from mercari_shopper.agent.progress import ProgressStep, ProgressTimeline

if TYPE_CHECKING:
    from rich.text import Text


class LiveProgressReporter:
    """Context manager + ProgressCallback. Streams a growing step timeline to stderr.

    Rich mode redraws the checklist in place. Plain mode prints each new line
    once. Both write only to the given stream, never stdout.
    """

    def __init__(self, stream: TextIO, *, plain: bool) -> None:
        self._stream = stream
        self._plain = plain
        self._timeline = ProgressTimeline()
        self._live: Any = None

    def __enter__(self) -> LiveProgressReporter:
        if not self._plain:
            from rich.console import Console
            from rich.live import Live

            console = Console(file=self._stream, highlight=False, soft_wrap=True)
            self._live = Live(self._renderable(), console=console, refresh_per_second=8)
            self._live.__enter__()
        return self

    def __call__(self, message: str) -> None:
        steps = self._timeline.advance(message)
        if self._live is not None:
            self._live.update(self._renderable(steps))
            return
        self._stream.write(message if message.endswith("\n") else message + "\n")
        flush = getattr(self._stream, "flush", None)
        if flush is not None:
            flush()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        steps = self._timeline.finish()
        live = self._live
        if live is not None:
            live.update(self._renderable(steps))
            live.__exit__(exc_type, exc, tb)
            self._live = None

    def _renderable(self, steps: tuple[ProgressStep, ...] | None = None) -> Text:
        from rich.text import Text

        items = steps if steps is not None else self._timeline.steps
        text = Text()
        for index, step in enumerate(items):
            if index:
                text.append("\n")
            mark = "✔" if step.done else "…"
            style = "dim" if step.done else "bold cyan"
            text.append(f"{mark} {step.text}", style=style)
        return text
