from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from mldoctor.core.schemas import (
    DatasetInfo,
    DiagnosticResult,
    ExecutionInfo,
    Finding,
    Report,
    RunInfo,
    Summary,
    build_summary,
    finite_or_none,
    jsonable,
)


def _finding(severity: str = "medium", evidence: dict | None = None) -> Finding:
    return Finding(
        code="HIGH_MISSINGNESS",
        severity=severity,  # type: ignore[arg-type]
        title="t",
        message="m",
        evidence=evidence or {"x": 1.0},
        recommendation="r",
    )


def _report(results: list[DiagnosticResult], summary: Summary | None = None) -> Report:
    now = datetime.now(UTC)
    return Report(
        run=RunInfo(
            id="x", task="regression", seed=1, started_at=now, completed_at=now,
            config_fingerprint="f", evaluation_target="unspecified",
        ),
        dataset=DatasetInfo(train_rows=1, feature_count=1, numeric_feature_count=1, target="y"),
        summary=summary or build_summary(results),
        diagnostics=results,
        execution=ExecutionInfo(duration_ms=1.0),
    )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_evidence_rejected(bad: float) -> None:
    with pytest.raises(ValidationError, match="non-finite"):
        _finding(evidence={"nested": {"values": [1.0, bad]}})


def test_non_finite_metrics_rejected() -> None:
    with pytest.raises(ValidationError):
        DiagnosticResult(id="a", status="passed", metrics={"r": float("nan")})


def test_skipped_requires_reason_and_no_findings() -> None:
    with pytest.raises(ValidationError, match="skip_reason"):
        DiagnosticResult(id="a", status="skipped")
    with pytest.raises(ValidationError):
        DiagnosticResult(id="a", status="passed", skip_reason="why")
    with pytest.raises(ValidationError):
        DiagnosticResult(id="a", status="error")


def test_finding_code_format_enforced() -> None:
    with pytest.raises(ValidationError):
        Finding(code="lower", severity="low", title="t", message="m", recommendation="r")


def test_summary_counts_must_match() -> None:
    results = [
        DiagnosticResult(id="a", status="warning", findings=[_finding("medium")]),
        DiagnosticResult(id="b", status="skipped", skip_reason="no test split"),
    ]
    report = _report(results)
    assert report.summary.warning == 1
    assert report.summary.skipped == 1
    assert report.summary.highest_severity == "medium"
    bad = Summary(checks_total=2, passed=2, warning=0, failed=0, skipped=0, error=0)
    with pytest.raises(ValidationError, match="summary counts"):
        _report(results, bad)


def test_summary_highest_severity_must_match() -> None:
    results = [DiagnosticResult(id="a", status="failed", findings=[_finding("critical")])]
    bad = Summary(checks_total=1, passed=0, warning=0, failed=1, skipped=0, error=0,
                  highest_severity="low")
    with pytest.raises(ValidationError, match="highest_severity"):
        _report(results, bad)


def test_naive_timestamp_rejected() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        RunInfo(id="x", task="regression", seed=1, started_at=datetime(2026, 1, 1),
                completed_at=datetime(2026, 1, 1), config_fingerprint="f",
                evaluation_target="unspecified")


def test_jsonable_and_finite_helpers() -> None:
    import numpy as np

    assert jsonable({"a": np.int64(3), "b": [np.float32(0.5)], "c": np.bool_(True)}) == {
        "a": 3, "b": [0.5], "c": True,
    }
    with pytest.raises(ValueError):
        jsonable(float("nan"))
    assert finite_or_none(float("inf")) is None
    assert finite_or_none(np.float64(2.0)) == 2.0
