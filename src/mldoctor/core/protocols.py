"""Diagnostic plugin protocol and a convenience base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

from mldoctor.core.context import DiagnosticContext
from mldoctor.core.schemas import DiagnosticResult, Finding, Task

Requirement = Literal["test", "test_target", "test_predictions", "train_predictions", "group"]

REQUIREMENT_REASONS: dict[str, str] = {
    "test": "no test/held-out split was supplied (--test)",
    "test_target": "the test split has no target column",
    "test_predictions": "no held-out predictions were supplied (--predictions)",
    "train_predictions": "no training-set predictions were supplied (--train-predictions)",
    "group": "no group column was declared (--group)",
}


@runtime_checkable
class Diagnostic(Protocol):
    """Contract every diagnostic (built-in or plugin) must satisfy."""

    @property
    def id(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def title(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def supported_tasks(self) -> frozenset[Task]: ...

    @property
    def requires(self) -> frozenset[Requirement]: ...

    def default_config(self) -> dict[str, Any]: ...

    def applicability(self, context: DiagnosticContext) -> str | None:
        """Return a human-readable skip reason, or ``None`` if the check can run."""
        ...

    def run(self, context: DiagnosticContext) -> DiagnosticResult: ...


def missing_requirements(context: DiagnosticContext, requires: frozenset[Requirement]) -> list[str]:
    available = {
        "test": context.test is not None,
        "test_target": context.test is not None and context.test_has_target,
        "test_predictions": context.test_predictions is not None,
        "train_predictions": context.train_predictions is not None,
        "group": context.group is not None,
    }
    return [REQUIREMENT_REASONS[r] for r in sorted(requires) if not available[r]]


class BaseDiagnostic(ABC):
    """Helper base class implementing applicability and result assembly."""

    id: ClassVar[str]
    version: ClassVar[str] = "0.1.0"
    title: ClassVar[str]
    description: ClassVar[str]
    supported_tasks: ClassVar[frozenset[Task]] = frozenset({"classification", "regression"})
    requires: ClassVar[frozenset[Requirement]] = frozenset()
    limitations: ClassVar[tuple[str, ...]] = ()

    def default_config(self) -> dict[str, Any]:
        return {}

    def applicability(self, context: DiagnosticContext) -> str | None:
        if context.task not in self.supported_tasks:
            return (
                f"not applicable to {context.task} tasks "
                f"(supports: {', '.join(sorted(self.supported_tasks))})"
            )
        missing = missing_requirements(context, self.requires)
        if missing:
            return "; ".join(missing)
        return None

    @abstractmethod
    def run(self, context: DiagnosticContext) -> DiagnosticResult: ...

    def result(
        self,
        findings: list[Finding],
        metrics: dict[str, Any],
        skipped_subchecks: dict[str, str] | None = None,
        extra_limitations: list[str] | None = None,
    ) -> DiagnosticResult:
        # Local import keeps the rules module free to depend on core.schemas only.
        from mldoctor.recommendations.rules import status_from_findings

        return DiagnosticResult(
            id=self.id,
            title=self.title,
            version=self.version,
            status=status_from_findings(findings),
            findings=findings,
            metrics=metrics,
            skipped_subchecks=skipped_subchecks or {},
            limitations=[*self.limitations, *(extra_limitations or [])],
        )
