"""Built-in diagnostics, in report order."""

from __future__ import annotations

from mldoctor.core.protocols import Diagnostic
from mldoctor.diagnostics.correlation import FeatureCorrelationDiagnostic
from mldoctor.diagnostics.data_integrity import DataIntegrityDiagnostic
from mldoctor.diagnostics.imbalance import ClassImbalanceDiagnostic
from mldoctor.diagnostics.outliers import OutlierDiagnostic


def builtin_diagnostics() -> list[Diagnostic]:
    return [
        DataIntegrityDiagnostic(),
        ClassImbalanceDiagnostic(),
        FeatureCorrelationDiagnostic(),
        OutlierDiagnostic(),
    ]
