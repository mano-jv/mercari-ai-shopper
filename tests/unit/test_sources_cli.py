import pytest
from mercari_shopper.sources.__main__ import main


def test_sources_cli_fixture_search(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--q", "switch 有機EL", "--max", "5", "--source", "fixture"]) == 0
    out = capsys.readouterr().out
    assert "m10000000001" in out
    assert "Nintendo Switch" in out
