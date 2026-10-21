from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd
import pytest

from mldoctor.api import build_context, diagnose
from mldoctor.core.context import DiagnosticContext
from mldoctor.core.engine import run_one
from mldoctor.core.protocols import BaseDiagnostic, Requirement
from mldoctor.core.registry import DiagnosticRegistry, RegistryError
from mldoctor.core.schemas import DiagnosticResult
from mldoctor.recommendations.rules import make_finding


class Boom(BaseDiagnostic):
    id = "boom"
    title = "Raises"
    description = "always raises"

    def run(self, context: DiagnosticContext) -> DiagnosticResult:
        raise RuntimeError("kaboom")


class NeedsTest(BaseDiagnostic):
    id = "needs_test"
    title = "Needs test"
    description = "requires test split"
    requires: ClassVar[frozenset[Requirement]] = frozenset({"test"})

    def run(self, context: DiagnosticContext) -> DiagnosticResult:
        return self.result([], {})


class Flags(BaseDiagnostic):
    id = "flags"
    title = "Flags"
    description = "emits findings"

    def run(self, context: DiagnosticContext) -> DiagnosticResult:
        findings = [
            make_finding("CONSTANT_FEATURE", "low", "b", {"n": 1}),
            make_finding("GROUP_OVERLAP", "critical", "a", {"n": 2}),
        ]
        metrics: dict[str, Any] = {"value": 1}
        return self.result(findings, metrics)


def _ctx() -> DiagnosticContext:
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    return build_context(train=train, target="y", task="regression")[0]


def test_exception_becomes_error_result() -> None:
    result = run_one(Boom(), _ctx())
    assert result.status == "error"
    assert "kaboom" in (result.error_message or "")
    assert result.findings == []


def test_missing_requirement_skips_with_reason() -> None:
    result = run_one(NeedsTest(), _ctx())
    assert result.status == "skipped"
    assert "test" in (result.skip_reason or "")


def test_findings_sorted_and_status_failed() -> None:
    result = run_one(Flags(), _ctx())
    assert [f.severity for f in result.findings] == ["critical", "low"]
    assert result.status == "failed"


def test_registry_rejects_duplicates_and_bad_ids() -> None:
    reg = DiagnosticRegistry()
    reg.register(Boom())
    with pytest.raises(RegistryError):
        reg.register(Boom())
    with pytest.raises(RegistryError):
        reg.select(["nope"])


def test_partial_report_counts_errors() -> None:
    reg = DiagnosticRegistry()
    for d in (Boom(), NeedsTest(), Flags()):
        reg.register(d)
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    report = diagnose(train=train, target="y", task="regression", registry=reg)
    assert (report.summary.error, report.summary.skipped, report.summary.failed) == (1, 1, 1)
    assert report.summary.highest_severity == "critical"
    assert [r.id for r in report.diagnostics] == ["boom", "needs_test", "flags"]


def test_rng_streams_are_reproducible_and_independent() -> None:
    ctx = _ctx()
    a1 = ctx.rng("uncertainty").integers(0, 1_000_000, 5)
    a2 = ctx.rng("uncertainty").integers(0, 1_000_000, 5)
    b = ctx.rng("other").integers(0, 1_000_000, 5)
    assert a1.tolist() == a2.tolist()
    assert a1.tolist() != b.tolist()


def test_plugins_load_only_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from mldoctor.core import registry as registry_module

    class _EP:
        name = "flags"

        def load(self) -> type:
            return Flags

    monkeypatch.setattr(registry_module, "entry_points", lambda group: [_EP()])
    assert "flags" not in registry_module.default_registry().ids()
    assert "flags" in registry_module.default_registry(load_plugins=True).ids()


def test_builtin_registry_order() -> None:
    from mldoctor.core.registry import default_registry

    assert default_registry().ids() == [
        "data_integrity", "class_imbalance", "feature_correlation", "outliers", "leakage",
        "overfitting", "uncertainty", "prediction_errors",
    ]
