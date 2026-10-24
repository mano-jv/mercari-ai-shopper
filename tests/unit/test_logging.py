import json

import pytest
from mercari_shopper.logging import configure_logging, log_event


def test_log_event_writes_json_line(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(verbose=True)
    log_event("search", keyword="switch", n=3)
    payload = json.loads(capsys.readouterr().err)
    assert payload["event"] == "search"
    assert payload["keyword"] == "switch"
    assert payload["n"] == 3
