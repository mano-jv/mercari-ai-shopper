import io
import json

import pytest
from mercari_shopper.agent.errors import AgentBudgetError
from mercari_shopper.agent.loop import AgentResult
from mercari_shopper.cli import build_parser, main
from mercari_shopper.domain.models import Recommendation
from mercari_shopper.presentation.live import LiveProgressReporter
from tests.unit.fakes import ErrorShopper, ImmediateShopper
from tests.unit.test_presenter import EXPECTED_PLAIN, _result


class _TtyIO(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_parser_flags() -> None:
    args = build_parser().parse_args(
        [
            "--provider",
            "openai",
            "--model",
            "gpt-4.1",
            "--offline",
            "--json",
            "--verbose",
            "--non-interactive",
            "-q",
            "switch",
        ]
    )
    assert args.provider == "openai"
    assert args.model == "gpt-4.1"
    assert args.offline is True
    assert args.json is True
    assert args.verbose is True
    assert args.non_interactive is True
    assert args.query == "switch"


def test_cli_prints_version() -> None:
    stdout = io.StringIO()
    assert main(["--version"], stdout=stdout) == 0
    assert stdout.getvalue() == "mercari-ai-shopper 0.1.0\n"


def test_cli_help_without_query() -> None:
    stderr = io.StringIO()
    assert main([], stdout=io.StringIO(), stderr=stderr) == 2
    assert "usage:" in stderr.getvalue().lower()


def test_cli_live_progress_goes_to_stderr() -> None:
    stdout = _TtyIO()
    stderr = io.StringIO()
    shopper = ImmediateShopper(
        _result(),
        progress_messages=("Searching Mercari…", "Found 3 listings"),
    )
    code = main(
        ["--plain", "-q", "switch"],
        shopper=shopper,
        stdout=stdout,
        stderr=stderr,
    )
    assert code == 0
    err = stderr.getvalue()
    assert "Searching Mercari" in err
    assert "Found 3 listings" in err
    out = stdout.getvalue()
    assert "Searching Mercari" not in out
    assert out == EXPECTED_PLAIN


def test_cli_progress_skipped_when_not_a_tty() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    shopper = ImmediateShopper(
        _result(),
        progress_messages=("Searching Mercari…",),
    )
    code = main(
        ["--plain", "-q", "switch"],
        shopper=shopper,
        stdout=stdout,
        stderr=stderr,
    )
    assert code == 0
    assert stderr.getvalue() == ""
    assert stdout.getvalue() == EXPECTED_PLAIN


def test_live_reporter_plain_writes_each_line_once() -> None:
    stream = io.StringIO()
    with LiveProgressReporter(stream, plain=True) as reporter:
        reporter("Searching Mercari…")
        reporter("Found 3 listings")
    assert stream.getvalue() == "Searching Mercari…\nFound 3 listings\n"


def test_cli_verbose_prints_step_trace() -> None:
    stdout = io.StringIO()
    code = main(
        ["--plain", "--verbose", "-q", "switch"],
        shopper=ImmediateShopper(_result()),
        stdout=stdout,
        stderr=io.StringIO(),
    )
    assert code == 0
    text = stdout.getvalue()
    assert "search_mercari" in text
    assert "get_item_details" in text
    assert "submit_recommendations" in text
    assert "1.23s" in text


def test_cli_one_shot_uses_injected_shopper() -> None:
    shopper = ImmediateShopper(_result())
    stdout = io.StringIO()
    code = main(
        ["--plain", "-q", "a used Nintendo Switch OLED under 30000 yen"],
        shopper=shopper,
        stdout=stdout,
        stderr=io.StringIO(),
    )
    assert code == 0
    assert shopper.requests == ["a used Nintendo Switch OLED under 30000 yen"]
    assert stdout.getvalue() == EXPECTED_PLAIN


def test_cli_positional_query() -> None:
    shopper = ImmediateShopper(_result())
    stdout = io.StringIO()
    assert main(["--plain", "find a switch"], shopper=shopper, stdout=stdout) == 0
    assert shopper.requests == ["find a switch"]


def test_cli_json_output() -> None:
    shopper = ImmediateShopper(_result())
    stdout = io.StringIO()
    assert main(["--json", "-q", "switch"], shopper=shopper, stdout=stdout) == 0
    payload = json.loads(stdout.getvalue())
    assert payload["recommendations"][0]["item_id"] == "m10000000001"
    assert payload["recommendations"][0]["price"] == 24800


def test_cli_agent_error_is_nonzero() -> None:
    shopper = ErrorShopper(AgentBudgetError("out of steps"))
    stderr = io.StringIO()
    assert main(["-q", "switch"], shopper=shopper, stdout=io.StringIO(), stderr=stderr) == 1
    assert "out of steps" in stderr.getvalue()


def test_cli_rejects_interactive_with_non_interactive() -> None:
    stderr = io.StringIO()
    assert main(["-i", "--non-interactive"], stdout=io.StringIO(), stderr=stderr) == 2
    assert "--non-interactive" in stderr.getvalue()


def test_cli_prints_clarifying_question() -> None:
    question = AgentResult(
        summary="",
        recommendations=(),
        listings=(),
        steps=1,
        tool_calls=0,
        repairs=0,
        question="What kind of gift?",
    )
    stdout = io.StringIO()
    code = main(
        ["--plain", "--non-interactive", "-q", "a nice gift"],
        shopper=ImmediateShopper(question),
        stdout=stdout,
        stderr=io.StringIO(),
    )
    assert code == 0
    assert stdout.getvalue() == "What kind of gift?\n"


def test_cli_rejects_eval_with_query() -> None:
    stderr = io.StringIO()
    assert main(["--eval", "-q", "switch"], stdout=io.StringIO(), stderr=stderr) == 2
    assert "--eval" in stderr.getvalue()


def test_cli_eval_offline_fixtures() -> None:
    stdout = io.StringIO()
    assert main(["--eval"], stdout=stdout, stderr=io.StringIO()) == 0
    text = stdout.getvalue()
    assert "10/10 passed" in text
    assert text.count("PASS") == 10


def test_cli_eval_json() -> None:
    stdout = io.StringIO()
    assert main(["--eval", "--json"], stdout=stdout, stderr=io.StringIO()) == 0
    payload = json.loads(stdout.getvalue())
    assert payload["ok"] is True
    assert payload["passed"] == 10
    assert len(payload["verdicts"]) == 10


def test_cli_prints_setup_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "mercari_shopper.cli.load_settings",
        lambda: (_ for _ in ()).throw(ValueError("Vertex AI needs google-auth")),
    )
    stderr = io.StringIO()
    assert main(["-q", "switch"], stdout=io.StringIO(), stderr=stderr) == 1
    assert "google-auth" in stderr.getvalue()


def test_cli_eval_fails_when_shopper_breaks_price_cap() -> None:
    over = AgentResult(
        summary="too expensive",
        recommendations=(
            Recommendation(item_id="m1", rank=1, reason="x"),
            Recommendation(item_id="m2", rank=2, reason="x"),
            Recommendation(item_id="m3", rank=3, reason="x"),
        ),
        listings=(
            {"item_id": "m1", "price": 99999, "condition_id": 3},
            {"item_id": "m2", "price": 99999, "condition_id": 3},
            {"item_id": "m3", "price": 99999, "condition_id": 3},
        ),
        steps=1,
        tool_calls=1,
        repairs=0,
    )
    stdout = io.StringIO()
    assert main(["--eval"], shopper=ImmediateShopper(over), stdout=stdout) == 1
    assert "FAIL" in stdout.getvalue()
    assert "exceeds cap" in stdout.getvalue()
