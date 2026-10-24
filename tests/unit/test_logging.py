import json
from typing import Any

import pytest
from mercari_shopper.logging import configure_logging, log_event


def log_events(err: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in err.splitlines():
        text = line.strip()
        if not text.startswith("{"):
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "event" in payload:
            events.append(payload)
    return events


def test_log_event_writes_json_line(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(verbose=True)
    log_event("search", keyword="switch", n=3)
    payload = json.loads(capsys.readouterr().err)
    assert payload["event"] == "search"
    assert payload["keyword"] == "switch"
    assert payload["n"] == 3
