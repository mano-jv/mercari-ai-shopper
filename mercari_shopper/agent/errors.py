"""Errors the shopping loop raises. Callers branch on the subclass."""


class AgentError(Exception):
    """The loop stopped without grounded recommendations."""


class AgentInputError(AgentError):
    """The user request is empty or longer than the input cap."""


class AgentBudgetError(AgentError):
    """A step, tool-call, or time limit was hit before submit_recommendations."""


class AgentGroundingError(AgentError):
    """submit_recommendations named item ids the tools never returned."""


class AgentLLMError(AgentError):
    """The provider failed, or retries of a retryable error ran out."""
