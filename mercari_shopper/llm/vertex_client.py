"""Claude on Vertex AI.

The request and response shape match AnthropicAdapter. Only the client
changes: AsyncAnthropicVertex signs in with Google Application Default
Credentials instead of ANTHROPIC_API_KEY.

Set GOOGLE_APPLICATION_CREDENTIALS to a service-account file, or run
`gcloud auth application-default login`. The default model is
claude-haiku-5-5. Pass LLM_MODEL or --model for Opus or Sonnet.

Credentials are loaded from the ADC file once per process and passed into
the SDK. That skips `google.auth.default()`, which shells out to `gcloud`
to look up a project id we already have from settings.
"""

from __future__ import annotations

import asyncio
import functools
import os
from time import monotonic
from typing import Any

from anthropic import AsyncAnthropicVertex

from mercari_shopper.config import DEFAULT_LLM_MAX_TOKENS
from mercari_shopper.llm.anthropic_client import AnthropicAdapter
from mercari_shopper.logging import log_event

_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


class AnthropicVertexAdapter(AnthropicAdapter):
    """Same Messages API as AnthropicAdapter, sent to a Vertex region."""

    def __init__(
        self,
        *,
        model: str,
        project_id: str = "",
        region: str = "",
        max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
        timeout: float = 60.0,
        client: Any | None = None,
        effort: str = "",
    ) -> None:
        owns_client = client is None
        project = project_id.strip()
        location = region.strip()
        if owns_client and (not project or not location):
            raise ValueError("Vertex AI needs a project id and a region")
        sdk: Any
        if owns_client:
            sdk = AsyncAnthropicVertex(
                project_id=project,
                region=location,
                timeout=timeout,
                credentials=_shared_credentials(),
            )
        else:
            sdk = client
        super().__init__(
            model=model,
            max_tokens=max_tokens,
            timeout=timeout,
            client=sdk,
            owns_client=owns_client,
            provider="vertex",
            effort=effort,
        )
        self.project_id = project
        self.region = location

    async def warm(self) -> None:
        """Refresh the ADC token before the first chat request."""
        if not self._owns_client:
            return
        credentials = getattr(self._client, "credentials", None)
        if credentials is None:
            return
        if getattr(credentials, "token", None) and not getattr(credentials, "expired", True):
            return
        refresh = getattr(credentials, "refresh", None)
        if not callable(refresh):
            return
        started = monotonic()
        await asyncio.to_thread(_refresh_credentials, credentials)
        log_event("llm_warm", provider=self.provider, elapsed_s=round(monotonic() - started, 3))


def _adc_file_path() -> str | None:
    env = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if env:
        return env
    return _well_known_adc_path()


def _well_known_adc_path() -> str | None:
    try:
        from google.auth import _cloud_sdk
    except ImportError:
        return None
    getter: Any = _cloud_sdk.get_application_default_credentials_path
    path = getter()
    if isinstance(path, str) and path and os.path.isfile(path):
        return path
    return None


@functools.cache
def _shared_credentials() -> Any:
    """One credentials object per process. Token refresh mutates it in place."""
    _require_google_auth()
    auth: Any
    import google.auth as auth

    path = _adc_file_path()
    if path is not None:
        credentials, _project = auth.load_credentials_from_file(
            path,
            scopes=[_CLOUD_PLATFORM_SCOPE],
        )
        return credentials
    credentials, _project = auth.default(scopes=[_CLOUD_PLATFORM_SCOPE])
    return credentials


def _refresh_credentials(credentials: Any) -> None:
    from google.auth.transport.requests import Request

    credentials.refresh(Request())


def _require_google_auth() -> None:
    try:
        import google.auth  # noqa: F401
    except ImportError as exc:
        raise ValueError(
            "Vertex AI needs google-auth. From the project venv run: "
            "pip install 'anthropic[vertex]' google-auth"
        ) from exc
