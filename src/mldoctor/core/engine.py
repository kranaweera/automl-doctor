"""Execution engine: runs diagnostics and assembles the report."""

from __future__ import annotations

import time
import traceback
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

from mldoctor.core.context import DiagnosticContext
from mldoctor.core.protocols import Diagnostic
from mldoctor.core.schemas import (
    DatasetInfo,
    DiagnosticResult,
    ExecutionInfo,
    InputFileInfo,
    Report,
    RunInfo,
    build_summary,
)
from mldoctor.recommendations.rules import sort_findings


def run_one(diagnostic: Diagnostic, context: DiagnosticContext) -> DiagnosticResult:
    """Run a single diagnostic; never raises.

    Inapplicable checks become ``skipped`` with a reason. Unexpected exceptions become
    ``error`` results so that one failing check cannot silently disappear or abort others.
    """
    start = time.perf_counter()
    title = getattr(diagnostic, "title", None)
    version = getattr(diagnostic, "version", None)
    try:
        reason = diagnostic.applicability(context)
        if reason is not None:
            result = DiagnosticResult(
                id=diagnostic.id, title=title, version=version, status="skipped",
                skip_reason=reason,
            )
        else:
            result = diagnostic.run(context)
            if result.id != diagnostic.id:
                raise ValueError(f"diagnostic returned result for '{result.id}'")
            result = result.model_copy(update={"findings": sort_findings(result.findings)})
            # Re-validate so plugin results obey the same invariants as built-ins.
            result = DiagnosticResult.model_validate(result.model_dump())
    except Exception as exc:
        last = traceback.extract_tb(exc.__traceback__)[-1] if exc.__traceback__ else None
        where = f" (at {last.filename.rsplit('/', 1)[-1]}:{last.lineno})" if last else ""
        result = DiagnosticResult(
            id=diagnostic.id, title=title, version=version, status="error",
            error_message=f"{type(exc).__name__}: {exc}{where}",
        )
    duration = (time.perf_counter() - start) * 1000.0
    return result.model_copy(update={"duration_ms": round(duration, 3)})


def run_diagnostics(
    diagnostics: Sequence[Diagnostic], context: DiagnosticContext
) -> list[DiagnosticResult]:
    return [run_one(d, context) for d in diagnostics]


def build_report(
    context: DiagnosticContext,
    results: list[DiagnosticResult],
    *,
    started_at: datetime,
    command: str,
    inputs: list[InputFileInfo],
    duration_ms: float,
) -> Report:
    dataset = DatasetInfo(
        train_rows=len(context.train),
        test_rows=None if context.test is None else len(context.test),
        feature_count=len(context.feature_columns),
        numeric_feature_count=len(context.numeric_features),
        target=context.target,
        row_id_column=context.row_id,
        group_column=context.group,
        has_test_predictions=context.test_predictions is not None,
        has_train_predictions=context.train_predictions is not None,
        class_labels=context.class_labels,
        inputs=inputs,
    )
    run = RunInfo(
        id=str(uuid.uuid4()),
        task=context.task,
        seed=context.seed,
        started_at=started_at,
        completed_at=datetime.now(UTC),
        command=command,
        config_fingerprint=context.config.fingerprint(),
        evaluation_target=context.config.evaluation_target,
    )
    return Report(
        run=run,
        dataset=dataset,
        summary=build_summary(results),
        diagnostics=results,
        execution=ExecutionInfo(duration_ms=round(duration_ms, 3), warnings=list(context.warnings)),
    )
