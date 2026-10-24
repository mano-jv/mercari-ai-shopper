from mercari_shopper.sources.errors import (
    MercariError,
    MercariHttpError,
    MercariParseError,
    parse_retry_after,
)


def test_http_error_exposes_status_and_retry_after() -> None:
    error = MercariHttpError(429, "slow down", retry_after=2.5)
    assert isinstance(error, MercariError)
    assert error.status == 429
    assert error.retry_after == 2.5
    assert "429" in str(error)
    assert "slow down" in str(error)


def test_parse_error_is_mercari_error() -> None:
    assert issubclass(MercariParseError, MercariError)


def test_parse_retry_after() -> None:
    assert parse_retry_after("2.5") == 2.5
    assert parse_retry_after("0") == 0.0
    assert parse_retry_after(None) is None
    assert parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") is None
