"""Errors raised by Mercari sources. Callers branch on the status code."""


class MercariError(Exception):
    """Base error for Mercari data access."""


class MercariHttpError(MercariError):
    """The API answered with an HTTP error.

    retry_after is set when the response carries a numeric Retry-After.
    AutoSource uses 401, 403 and 429 as the signal to switch to the browser.
    """

    def __init__(self, status: int, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(f"Mercari HTTP {status}: {message}")
        self.status = status
        self.retry_after = retry_after


class MercariTransportError(MercariError):
    """The connection failed before a response arrived."""


class MercariParseError(MercariError):
    """The body was not the JSON object the parser expects."""


def parse_retry_after(value: str | None) -> float | None:
    """Return seconds from a numeric Retry-After header, else None."""
    if value is None or value.strip() == "":
        return None
    try:
        parsed = float(value.strip())
    except ValueError:
        return None
    return parsed if parsed >= 0 else None
