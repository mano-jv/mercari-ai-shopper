"""Errors from an LLM provider. The agent loop retries only retryable ones."""


class LLMError(Exception):
    """A provider call failed or returned a body this adapter cannot read."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status = status
        self.retryable = retryable


_RETRYABLE_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})


def translate_provider_error(exc: Exception, *, provider: str) -> LLMError:
    """Map an SDK exception to LLMError.

    Retry timeouts, connection failures, and 408/409/429/5xx. A programming
    error or a 4xx other than 429 stays non-retryable.
    """
    status = _status_code(exc)
    retryable = _is_transient(exc) if status is None else status in _RETRYABLE_STATUSES
    return LLMError(
        f"{provider} request failed: {exc}",
        provider=provider,
        status=status,
        retryable=retryable,
    )


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    name = type(exc).__name__.casefold()
    return any(token in name for token in ("timeout", "connect", "transport"))


def _status_code(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status
    return None
