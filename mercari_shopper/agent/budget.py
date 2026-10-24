"""Policy object for step, tool-call, and wall-clock limits."""

from __future__ import annotations

from dataclasses import dataclass

from mercari_shopper.agent.errors import AgentBudgetError
from mercari_shopper.config import Settings


@dataclass(frozen=True)
class AgentBudget:
    """Hard caps for one `ShopperAgent.run`. Terminal submit calls are not capped."""

    max_steps: int = 8
    max_tool_calls: int = 12
    timeout_seconds: float = 90.0

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if self.max_tool_calls < 1:
            raise ValueError("max_tool_calls must be at least 1")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

    @classmethod
    def from_settings(cls, settings: Settings) -> AgentBudget:
        return cls(
            max_steps=settings.agent_max_steps,
            max_tool_calls=settings.agent_max_tool_calls,
            timeout_seconds=settings.agent_timeout,
        )

    def check_turn(self, *, steps: int, elapsed: float) -> None:
        if elapsed >= self.timeout_seconds:
            raise AgentBudgetError(f"time budget exhausted ({self.timeout_seconds:g}s)")
        if steps >= self.max_steps:
            raise AgentBudgetError(f"step budget exhausted ({self.max_steps})")

    def check_tools(self, current: int, extra: int) -> None:
        if extra <= 0:
            return
        if current + extra > self.max_tool_calls:
            raise AgentBudgetError(f"tool-call budget exhausted ({self.max_tool_calls})")
