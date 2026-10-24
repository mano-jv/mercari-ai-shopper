"""Claude on Vertex AI.

The request and response shape match AnthropicAdapter. Only the client
changes: AsyncAnthropicVertex signs in with Google Application Default
Credentials instead of ANTHROPIC_API_KEY.

Set GOOGLE_APPLICATION_CREDENTIALS to a service-account file, or run
`gcloud auth application-default login`. The default model is
claude-haiku-5-5. Pass LLM_MODEL or --model for Opus or Sonnet.
"""

from __future__ import annotations

from typing import Any

from anthropic import AsyncAnthropicVertex

from mercari_shopper.config import DEFAULT_LLM_MAX_TOKENS
from mercari_shopper.llm.anthropic_client import AnthropicAdapter


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
    ) -> None:
        owns_client = client is None
        project = project_id.strip()
        location = region.strip()
        if owns_client and (not project or not location):
            raise ValueError("Vertex AI needs a project id and a region")
        if owns_client:
            _require_google_auth()
        sdk = (
            client
            if client is not None
            else AsyncAnthropicVertex(project_id=project, region=location, timeout=timeout)
        )
        super().__init__(
            model=model,
            max_tokens=max_tokens,
            timeout=timeout,
            client=sdk,
            owns_client=owns_client,
            provider="vertex",
        )
        self.project_id = project
        self.region = location


def _require_google_auth() -> None:
    try:
        import google.auth  # noqa: F401
    except ImportError as exc:
        raise ValueError(
            "Vertex AI needs google-auth. From the project venv run: "
            "pip install 'anthropic[vertex]' google-auth"
        ) from exc
