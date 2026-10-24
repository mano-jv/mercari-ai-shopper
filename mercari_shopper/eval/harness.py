"""Run golden cases against any Shopper and collect constraint verdicts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from mercari_shopper.agent.loop import AgentResult, Shopper
from mercari_shopper.agent.progress import STATUS_START, ProgressCallback, emit_progress
from mercari_shopper.domain.models import Listing, Recommendation
from mercari_shopper.eval.cases import GOLDEN_CASES, GoldenCase, validate_cases
from mercari_shopper.sources.base import MercariSource
from mercari_shopper.tools.compact import ResultCompactor


class Ranker(Protocol):
    """Strategy: choose up to n listings from a catalog page."""

    def pick(self, items: Sequence[Listing], n: int) -> tuple[Listing, ...]:
        """Stable ordering. Fewer than n items is allowed."""


class CheapestRanker:
    """Deterministic offline stand-in for the LLM ranker."""

    def pick(self, items: Sequence[Listing], n: int) -> tuple[Listing, ...]:
        ordered = sorted(items, key=lambda item: (item.price, item.item_id))
        return tuple(ordered[:n])


class CatalogShopper:
    """Adapter: MercariSource + golden-case filters → Shopper, no LLM."""

    def __init__(
        self,
        source: MercariSource,
        cases: Sequence[GoldenCase],
        *,
        ranker: Ranker | None = None,
        compactor: ResultCompactor | None = None,
    ) -> None:
        self._source = source
        self._cases = {case.query: case for case in validate_cases(cases)}
        self._ranker = ranker or CheapestRanker()
        self._compactor = compactor or ResultCompactor()

    async def run(
        self,
        request: str,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> AgentResult:
        emit_progress(on_progress, STATUS_START)
        case = self._cases.get(request)
        if case is None:
            raise KeyError(f"no golden case for {request!r}")
        found = await self._source.search(case.to_search_query())
        picks = self._ranker.pick(found.items, n=3)
        recommendations = tuple(
            Recommendation(
                item_id=item.item_id,
                rank=index,
                reason=f"lowest matching price at ¥{item.price}",
                pros=["matches eval filters"],
                cons=[],
            )
            for index, item in enumerate(picks, start=1)
        )
        listings = tuple(self._compactor.compact_listing(item) for item in picks)
        return AgentResult(
            summary=f"{len(picks)} listing(s) for {case.id}",
            recommendations=recommendations,
            listings=listings,
            steps=1,
            tool_calls=1,
            repairs=0,
        )


@dataclass(frozen=True)
class CaseVerdict:
    case_id: str
    query: str
    passed: bool
    failures: tuple[str, ...]
    result_count: int


@dataclass(frozen=True)
class EvalReport:
    verdicts: tuple[CaseVerdict, ...]

    @property
    def passed(self) -> int:
        return sum(1 for verdict in self.verdicts if verdict.passed)

    @property
    def failed(self) -> int:
        return len(self.verdicts) - self.passed

    @property
    def ok(self) -> bool:
        return self.failed == 0

    def to_payload(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "passed": self.passed,
            "failed": self.failed,
            "verdicts": [
                {
                    "case_id": verdict.case_id,
                    "query": verdict.query,
                    "passed": verdict.passed,
                    "failures": list(verdict.failures),
                    "result_count": verdict.result_count,
                }
                for verdict in self.verdicts
            ],
        }

    def render(self) -> str:
        lines = [
            f"{'PASS' if verdict.passed else 'FAIL'}  {verdict.case_id}"
            for verdict in self.verdicts
        ]
        for verdict in self.verdicts:
            if verdict.passed:
                continue
            for failure in verdict.failures:
                lines.append(f"  - {failure}")
        lines.append(f"{self.passed}/{len(self.verdicts)} passed")
        return "\n".join(lines) + "\n"


class EvalHarness:
    """Template: run each case, apply its constraint strategies, collect a report."""

    def __init__(self, cases: Sequence[GoldenCase] | None = None) -> None:
        self._cases = validate_cases(cases or GOLDEN_CASES)

    @property
    def cases(self) -> tuple[GoldenCase, ...]:
        return self._cases

    async def run(self, shopper: Shopper) -> EvalReport:
        verdicts: list[CaseVerdict] = []
        for case in self._cases:
            try:
                result = await shopper.run(case.query)
            except Exception as exc:
                verdicts.append(
                    CaseVerdict(
                        case_id=case.id,
                        query=case.query,
                        passed=False,
                        failures=(f"raised {type(exc).__name__}: {exc}",),
                        result_count=0,
                    )
                )
                continue
            failures: list[str] = []
            for constraint in case.constraints():
                failures.extend(constraint.check(result))
            verdicts.append(
                CaseVerdict(
                    case_id=case.id,
                    query=case.query,
                    passed=not failures,
                    failures=tuple(failures),
                    result_count=len(result.recommendations),
                )
            )
        return EvalReport(tuple(verdicts))
