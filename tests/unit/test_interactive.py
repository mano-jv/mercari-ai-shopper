import asyncio
import io

from mercari_shopper.agent.errors import AgentInputError
from mercari_shopper.cli import main
from mercari_shopper.presentation.interactive import InteractiveSession, Stdio
from mercari_shopper.presentation.render import PlainPresenter, RichPresenter
from tests.unit.fakes import ErrorShopper, ImmediateShopper
from tests.unit.test_presenter import _result


class _TtyIO(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_interactive_runs_then_quits() -> None:
    shopper = ImmediateShopper(_result())
    stdin = io.StringIO("find a switch\nquit\n")
    stdout = io.StringIO()
    code = main(
        ["-i", "--plain"],
        shopper=shopper,
        stdin=stdin,
        stdout=stdout,
        stderr=io.StringIO(),
    )
    assert code == 0
    assert shopper.requests == ["find a switch"]
    out = stdout.getvalue()
    assert "Mercari Japan shopper" in out
    assert "Nintendo Switch 有機EL 本体" in out


def test_interactive_help_empty_and_eof() -> None:
    shopper = ImmediateShopper(_result())
    stdin = io.StringIO("\nhelp\n")
    stdout = io.StringIO()
    assert main(["--interactive", "--plain"], shopper=shopper, stdin=stdin, stdout=stdout) == 0
    assert shopper.requests == []
    assert "Commands:" in stdout.getvalue()


def test_interactive_session_writes_agent_errors() -> None:
    stderr = io.StringIO()
    session = InteractiveSession(
        ErrorShopper(AgentInputError("too long")),
        PlainPresenter(),
        Stdio(io.StringIO("vague\nquit\n"), io.StringIO(), stderr),
    )
    assert asyncio.run(session.run()) == 0
    assert "too long" in stderr.getvalue()


def test_interactive_session_emits_through_presenter() -> None:
    stdout = io.StringIO()
    session = InteractiveSession(
        ImmediateShopper(_result()),
        RichPresenter(),
        Stdio(io.StringIO("find a switch\nquit\n"), stdout),
    )
    assert asyncio.run(session.run()) == 0
    assert "Mercari shopper" in stdout.getvalue()


def test_interactive_live_progress_goes_to_stderr() -> None:
    shopper = ImmediateShopper(
        _result(),
        progress_messages=("Searching Mercari…", "Found 3 listings"),
    )
    stdin = io.StringIO("find a switch\nquit\n")
    stdout = _TtyIO()
    stderr = io.StringIO()
    code = main(
        ["-i", "--plain"],
        shopper=shopper,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
    )
    assert code == 0
    err = stderr.getvalue()
    assert "Searching Mercari" in err
    assert "Found 3 listings" in err
    out = stdout.getvalue()
    assert "Searching Mercari" not in out
    assert "Nintendo Switch 有機EL 本体" in out


def test_cli_rejects_interactive_with_query() -> None:
    stderr = io.StringIO()
    assert main(["-i", "-q", "switch"], stdout=io.StringIO(), stderr=stderr) == 2
    assert "--interactive" in stderr.getvalue()
