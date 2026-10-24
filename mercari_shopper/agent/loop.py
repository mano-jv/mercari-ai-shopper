"""Hand-written tool-calling loop. Depends on LLMClient and ToolRegistry ports."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from time import monotonic
from typing import Any, Protocol, runtime_checkable

from mercari_shopper.agent.budget import AgentBudget
from mercari_shopper.agent.enrich import ListingEnricher, NoOpEnricher
from mercari_shopper.agent.errors import AgentGroundingError
from mercari_shopper.agent.guardrails import (
    GroundingReport,
    GroundingValidator,
    InputGuard,
    wrap_untrusted,
)
from mercari_shopper.agent.prompts import PromptBuilder
from mercari_shopper.agent.retry import retry_llm
from mercari_shopper.domain.models import Recommendation
from mercari_shopper.llm.base import (
    ForcedTool,
    LLMClient,
    LLMResponse,
    Message,
    Role,
    ToolCall,
    ToolChoice,
    ToolChoiceMode,
    ToolResult,
)
from mercari_shopper.logging import log_event
from mercari_shopper.tools.registry import ToolExecution, ToolRegistry, dump_json
from mercari_shopper.tools.schemas import SUBMIT_TOOL


@dataclass(frozen=True)
class StepTrace:
    """One LLM turn: wall time, LLM vs tools split, and per-tool elapsed."""

    step: int
    elapsed_s: float
    tool_names: tuple[str, ...]
    llm_s: float = 0.0
    tools_s: float = 0.0
    tool_elapsed_s: tuple[float, ...] = ()


@dataclass(frozen=True)
class AgentResult:
    """Grounded top-3. `listings` is compact tool data, aligned with rank 1..3.

    When the request was too vague, `question` is set and recommendations are empty.
    """

    summary: str
    recommendations: tuple[Recommendation, ...]
    listings: tuple[dict[str, Any], ...]
    steps: int
    tool_calls: int
    repairs: int
    elapsed_s: float = 0.0
    trace: tuple[StepTrace, ...] = ()
    question: str | None = None


@runtime_checkable
class Shopper(Protocol):
    """Port the CLI, interactive session, and eval harness share."""

    async def run(self, request: str) -> AgentResult:
        """Turn a natural-language request into grounded recommendations."""


class ShopperAgent:
    """Template: complete → run tools → validate submit → repair once or loop."""

    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        *,
        budget: AgentBudget | None = None,
        prompts: PromptBuilder | None = None,
        input_guard: InputGuard | None = None,
        validator: GroundingValidator | None = None,
        enricher: ListingEnricher | None = None,
        llm_attempts: int = 3,
        llm_retry_delay: float = 0.4,
        max_repairs: int = 1,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if llm_attempts < 1:
            raise ValueError("llm_attempts must be at least 1")
        if max_repairs < 0:
            raise ValueError("max_repairs must be at least 0")
        self._llm = llm
        self._registry = registry
        self._budget = budget or AgentBudget()
        self._prompts = prompts or PromptBuilder()
        self._input_guard = input_guard or InputGuard()
        self._validator = validator or GroundingValidator()
        self._enricher = enricher or NoOpEnricher()
        self._llm_attempts = llm_attempts
        self._llm_retry_delay = llm_retry_delay
        self._max_repairs = max_repairs
        self._sleep = sleep or asyncio.sleep
        self._clock = clock

    @property
    def budget(self) -> AgentBudget:
        return self._budget

    async def run(self, request: str) -> AgentResult:
        text = self._input_guard.sanitize(request)
        self._validator.reset()
        started = self._clock()
        log_event("agent_start", chars=len(text))
        messages: list[Message] = [
            Message(role=Role.SYSTEM, content=self._prompts.system()),
            Message(role=Role.USER, content=text),
        ]
        steps = 0
        tool_calls = 0
        repairs = 0
        force_submit = False
        traces: list[StepTrace] = []

        while True:
            self._budget.check_turn(steps=steps, elapsed=self._clock() - started)
            if self._should_force_submit(steps) and not force_submit:
                force_submit = True
                messages.append(Message(role=Role.USER, content=self._prompts.nudge_submit()))
                log_event("agent_force_submit", known_ids=len(self._validator.ids), step=steps + 1)
            log_event(
                "agent_step",
                step=steps + 1,
                tool_calls=tool_calls,
                repairs=repairs,
                force_submit=force_submit,
            )
            choice: ToolChoice = ForcedTool(SUBMIT_TOOL) if force_submit else ToolChoiceMode.AUTO
            step_started = self._clock()
            response = await self._complete(messages, choice)
            llm_done = self._clock()
            steps += 1
            messages.append(response.message)

            if not response.tool_calls:
                traces.append(
                    _step_trace(
                        steps,
                        step_started,
                        self._clock(),
                        (),
                        llm_s=llm_done - step_started,
                    )
                )
                messages.append(Message(role=Role.USER, content=self._prompts.nudge_tools()))
                continue

            executions, submit_data, question, grounding_failed, added = await self._apply_tools(
                response.tool_calls, tool_calls
            )
            tool_calls += added
            traces.append(
                _step_trace(
                    steps,
                    step_started,
                    self._clock(),
                    tuple(ex.name for ex in executions),
                    llm_s=llm_done - step_started,
                    tool_elapsed_s=tuple(ex.elapsed_s for ex in executions),
                )
            )
            if submit_data is not None:
                result = await self._to_result(
                    submit_data,
                    steps,
                    tool_calls,
                    repairs,
                    elapsed_s=round(self._clock() - started, 3),
                    trace=tuple(traces),
                )
                log_event(
                    "agent_complete",
                    steps=result.steps,
                    tool_calls=result.tool_calls,
                    repairs=result.repairs,
                    elapsed_s=result.elapsed_s,
                )
                return result
            if question is not None:
                log_event("agent_clarify", step=steps, question_chars=len(question))
                return AgentResult(
                    summary="",
                    recommendations=(),
                    listings=(),
                    steps=steps,
                    tool_calls=tool_calls,
                    repairs=repairs,
                    elapsed_s=round(self._clock() - started, 3),
                    trace=tuple(traces),
                    question=question,
                )

            if grounding_failed:
                if repairs >= self._max_repairs:
                    raise AgentGroundingError(
                        "recommendations used item ids that were not in tool results"
                    )
                repairs += 1
                force_submit = True
                log_event("agent_repair", repair=repairs)

            for execution in executions:
                messages.append(_tool_message(execution))

    def _should_force_submit(self, steps: int) -> bool:
        """Stop extra searches: last step, or three ids already in the catalog."""
        known = len(self._validator.ids)
        if known == 0:
            return False
        last_step = steps >= self._budget.max_steps - 1
        enough_ids = known >= 3 and steps >= 2
        return last_step or enough_ids

    async def _complete(self, messages: Sequence[Message], tool_choice: ToolChoice) -> LLMResponse:
        specs = self._registry.specs()

        async def once() -> LLMResponse:
            return await self._llm.complete(messages, tools=specs, tool_choice=tool_choice)

        return await retry_llm(
            once,
            attempts=self._llm_attempts,
            base_delay=self._llm_retry_delay,
            sleep=self._sleep,
        )

    async def _apply_tools(
        self,
        calls: tuple[ToolCall, ...],
        tool_calls_so_far: int,
    ) -> tuple[list[ToolExecution], object | None, str | None, bool, int]:
        others = [call for call in calls if not self._registry.is_terminal(call.name)]
        terminals = [call for call in calls if self._registry.is_terminal(call.name)]
        self._budget.check_tools(tool_calls_so_far, len(others))

        by_id: dict[str, ToolExecution] = {}
        if others:
            for execution in await self._registry.execute_many(others):
                by_id[execution.tool_call_id] = execution
                self._validator.observe(execution)

        submit_data: object | None = None
        question: str | None = None
        grounding_failed = False
        for call in terminals:
            execution = await self._registry.execute(call)
            if execution.is_error:
                by_id[call.id] = execution
                continue
            if self._registry.is_clarify(call.name):
                if self._validator.ids:
                    by_id[call.id] = _clarify_skipped(call, elapsed_s=execution.elapsed_s)
                    continue
                if question is None:
                    question = _question_text(execution.data)
                by_id[call.id] = execution
                continue
            if submit_data is None:
                report = self._validator.check_submit(execution.data)
                if report.ok:
                    submit_data = execution.data
                    by_id[call.id] = execution
                    continue
                grounding_failed = True
                execution = _ungrounded(call, report, elapsed_s=execution.elapsed_s)
            by_id[call.id] = execution

        ordered = [by_id[call.id] for call in calls if call.id in by_id]
        if submit_data is not None:
            question = None
        return ordered, submit_data, question, grounding_failed, len(others)

    async def _to_result(
        self,
        data: object,
        steps: int,
        tool_calls: int,
        repairs: int,
        *,
        elapsed_s: float,
        trace: tuple[StepTrace, ...],
    ) -> AgentResult:
        payload = data if isinstance(data, dict) else {}
        summary = payload.get("summary")
        rows = payload.get("recommendations")
        parsed = (
            [Recommendation.model_validate(row) for row in rows] if isinstance(rows, list) else []
        )
        recommendations = tuple(sorted(parsed, key=lambda rec: rec.rank))
        listings = tuple(
            self._validator.listing(rec.item_id) or {"item_id": rec.item_id}
            for rec in recommendations
        )
        listings = await self._enricher.enrich(listings, snapshot=self._validator.market)
        return AgentResult(
            summary=summary if isinstance(summary, str) else "",
            recommendations=recommendations,
            listings=listings,
            steps=steps,
            tool_calls=tool_calls,
            repairs=repairs,
            elapsed_s=elapsed_s,
            trace=trace,
        )


def _step_trace(
    step: int,
    started: float,
    now: float,
    tool_names: tuple[str, ...],
    *,
    llm_s: float = 0.0,
    tool_elapsed_s: tuple[float, ...] = (),
) -> StepTrace:
    elapsed = round(now - started, 3)
    llm = round(llm_s, 3)
    return StepTrace(
        step=step,
        elapsed_s=elapsed,
        tool_names=tool_names,
        llm_s=llm,
        tools_s=round(max(0.0, elapsed - llm), 3),
        tool_elapsed_s=tool_elapsed_s,
    )


def _tool_message(execution: ToolExecution) -> Message:
    return Message(
        role=Role.TOOL,
        tool_result=ToolResult(
            tool_call_id=execution.tool_call_id,
            content=wrap_untrusted(execution.content),
            is_error=execution.is_error,
        ),
    )


def _question_text(data: object) -> str:
    if isinstance(data, dict):
        question = data.get("question")
        if isinstance(question, str) and question.strip():
            return question.strip()
    return "Could you be more specific about what you want to buy?"


def _clarify_skipped(call: ToolCall, *, elapsed_s: float = 0.0) -> ToolExecution:
    return ToolExecution(
        tool_call_id=call.id,
        name=call.name,
        content=dump_json(
            {
                "error": "clarifying question skipped",
                "detail": (
                    "Listings are already in tool results. "
                    "Call submit_recommendations instead of ask_user."
                ),
            }
        ),
        is_error=True,
        elapsed_s=elapsed_s,
    )


def _ungrounded(
    call: ToolCall, report: GroundingReport, *, elapsed_s: float = 0.0
) -> ToolExecution:
    return ToolExecution(
        tool_call_id=call.id,
        name=call.name,
        content=dump_json(
            {
                "error": "ungrounded recommendations",
                "detail": report.message,
                "unknown_ids": list(report.unknown_ids),
                "known_ids": list(report.known_ids),
            }
        ),
        is_error=True,
        elapsed_s=elapsed_s,
    )
