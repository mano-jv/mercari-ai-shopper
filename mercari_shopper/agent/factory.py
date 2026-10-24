"""Compose a ShopperAgent from ports and settings. Callers own LLM/source lifecycle."""

from __future__ import annotations

from mercari_shopper.agent.budget import AgentBudget
from mercari_shopper.agent.guardrails import InputGuard
from mercari_shopper.agent.loop import ShopperAgent
from mercari_shopper.agent.prompts import PromptBuilder
from mercari_shopper.config import Settings, load_settings
from mercari_shopper.llm.base import LLMClient
from mercari_shopper.tools.registry import ToolRegistry


def build_agent(
    llm: LLMClient,
    registry: ToolRegistry,
    *,
    settings: Settings | None = None,
    budget: AgentBudget | None = None,
    prompts: PromptBuilder | None = None,
) -> ShopperAgent:
    """Factory: wire budget and input cap from Settings onto the loop."""
    resolved = settings or load_settings()
    return ShopperAgent(
        llm,
        registry,
        budget=budget or AgentBudget.from_settings(resolved),
        prompts=prompts or PromptBuilder(),
        input_guard=InputGuard(max_chars=resolved.max_user_chars),
        llm_attempts=resolved.agent_llm_attempts,
    )
