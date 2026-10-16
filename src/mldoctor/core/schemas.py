"""Versioned, strict report schemas.

All numeric evidence must be JSON-safe: ``NaN`` and ``±Infinity`` are rejected. Undefined
quantities are represented as ``null`` together with an explicit reason elsewhere in the
evidence (conventionally a ``*_undefined_reason`` key or an ``undefined`` mapping).
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from mldoctor._version import SCHEMA_VERSION, TOOL_NAME, __version__

Severity = Literal["info", "low", "medium", "high", "critical"]
Status = Literal["passed", "warning", "failed", "skipped", "error"]
Task = Literal["classification", "regression"]

SEVERITY_ORDER: dict[str, int] = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
STATUS_VALUES: tuple[Status, ...] = ("passed", "warning", "failed", "skipped", "error")


def _assert_finite(value: Any, path: str = "") -> None:
    """Recursively reject non-finite floats inside a JSON-like value."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                f"non-finite number at '{path or '<root>'}'; use null plus an explicit reason"
            )
    elif isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, f"{path}[{index}]")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)


class Finding(_Strict):
    """A single interpretable observation produced by a diagnostic."""

    code: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    severity: Severity
    title: str
    message: str
    evidence: dict[str, JsonValue] = Field(default_factory=dict)
    recommendation: str
    limitations: list[str] = Field(default_factory=list)

    @field_validator("evidence")
    @classmethod
    def _finite_evidence(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        _assert_finite(value, "evidence")
        return value


class DiagnosticResult(_Strict):
    """Outcome of one diagnostic.

    ``status`` semantics:

    * ``passed``  – all applicable checks ran and no configured alert fired.
    * ``warning`` – at least one low/medium finding.
    * ``failed``  – at least one high/critical finding (a real statistical finding,
      not a software failure).
    * ``skipped`` – the diagnostic was not applicable; ``skip_reason`` is mandatory.
    * ``error``   – an unexpected exception occurred; ``error_message`` is mandatory.
    """

    id: str
    status: Status
    findings: list[Finding] = Field(default_factory=list)
    metrics: dict[str, JsonValue] = Field(default_factory=dict)
    skip_reason: str | None = None
    # Extensions to the spec's minimal schema (documented in docs/report_schema.md).
    title: str | None = None
    version: str | None = None
    skipped_subchecks: dict[str, str] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    error_message: str | None = None
    duration_ms: float | None = Field(default=None, ge=0)

    @field_validator("metrics")
    @classmethod
    def _finite_metrics(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        _assert_finite(value, "metrics")
        return value

    @model_validator(mode="after")
    def _status_consistency(self) -> DiagnosticResult:
        if self.status == "skipped" and not self.skip_reason:
            raise ValueError(f"diagnostic '{self.id}' is skipped but has no skip_reason")
        if self.status != "skipped" and self.skip_reason is not None:
            raise ValueError(f"diagnostic '{self.id}' has skip_reason but status {self.status}")
        if self.status == "error" and not self.error_message:
            raise ValueError(f"diagnostic '{self.id}' has status error but no error_message")
        if self.status in ("skipped", "error") and self.findings:
            raise ValueError(f"diagnostic '{self.id}' with status {self.status} has findings")
        return self


class ToolInfo(_Strict):
    name: str = TOOL_NAME
    version: str = __version__


class RunInfo(_Strict):
    id: str
    task: Task
    seed: int
    started_at: datetime
    completed_at: datetime
    command: str = "diagnose"
    config_fingerprint: str
    evaluation_target: str

    @field_validator("started_at", "completed_at")
    @classmethod
    def _tz_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must be timezone-aware (UTC)")
        return value


class InputFileInfo(_Strict):
    role: str
    sha256: str | None = None
    rows: int = Field(ge=0)
    columns: int = Field(ge=0)


class DatasetInfo(_Strict):
    train_rows: int = Field(ge=0)
    test_rows: int | None = Field(default=None, ge=0)
    feature_count: int = Field(ge=0)
    numeric_feature_count: int = Field(ge=0)
    target: str
    row_id_column: str | None = None
    group_column: str | None = None
    has_test_predictions: bool = False
    has_train_predictions: bool = False
    class_labels: list[str] | None = None
    inputs: list[InputFileInfo] = Field(default_factory=list)


class Summary(_Strict):
    checks_total: int = Field(ge=0)
    passed: int = Field(ge=0)
    warning: int = Field(ge=0)
    failed: int = Field(ge=0)
    skipped: int = Field(ge=0)
    error: int = Field(ge=0)
    highest_severity: Severity | None = None
    findings_by_severity: dict[str, int] = Field(default_factory=dict)


class ExecutionInfo(_Strict):
    duration_ms: float = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)


class Report(_Strict):
    """Top-level report document (``schema_version`` 0.1.0)."""

    schema_version: str = SCHEMA_VERSION
    tool: ToolInfo = Field(default_factory=ToolInfo)
    run: RunInfo
    dataset: DatasetInfo
    summary: Summary
    diagnostics: list[DiagnosticResult]
    execution: ExecutionInfo

    @model_validator(mode="after")
    def _summary_matches(self) -> Report:
        counts = dict.fromkeys(STATUS_VALUES, 0)
        for result in self.diagnostics:
            counts[result.status] += 1
        expected = {"checks_total": len(self.diagnostics), **counts}
        actual = {
            "checks_total": self.summary.checks_total,
            "passed": self.summary.passed,
            "warning": self.summary.warning,
            "failed": self.summary.failed,
            "skipped": self.summary.skipped,
            "error": self.summary.error,
        }
        if expected != actual:
            raise ValueError(f"summary counts {actual} do not match diagnostics {expected}")
        severities = [f.severity for r in self.diagnostics for f in r.findings]
        highest = max(severities, key=SEVERITY_ORDER.__getitem__) if severities else None
        if highest != self.summary.highest_severity:
            raise ValueError(
                f"summary.highest_severity={self.summary.highest_severity} but findings imply "
                f"{highest}"
            )
        ids = [r.id for r in self.diagnostics]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate diagnostic ids in report")
        return self


def build_summary(results: list[DiagnosticResult]) -> Summary:
    """Derive the summary block from diagnostic results."""
    counts = dict.fromkeys(STATUS_VALUES, 0)
    by_severity = dict.fromkeys(SEVERITY_ORDER, 0)
    for result in results:
        counts[result.status] += 1
        for finding in result.findings:
            by_severity[finding.severity] += 1
    severities = [f.severity for r in results for f in r.findings]
    highest = max(severities, key=SEVERITY_ORDER.__getitem__) if severities else None
    return Summary(
        checks_total=len(results),
        highest_severity=highest,
        findings_by_severity=by_severity,
        **counts,
    )


def finite_or_none(value: Any) -> float | None:
    """Return ``float(value)`` if finite, else ``None`` (caller must record why)."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def jsonable(value: Any) -> Any:
    """Convert numpy/pandas scalars and containers to plain JSON types.

    Non-finite floats raise ``ValueError``: undefined quantities must be converted to
    ``None`` deliberately (see :func:`finite_or_none`) so that a reason can be attached.
    """
    import numpy as np  # local import keeps schema import light

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("non-finite value reached serialization; use finite_or_none()")
        return number
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    raise TypeError(f"value of type {type(value).__name__} is not JSON-serializable")
