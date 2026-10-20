"""Built-in diagnostics, in report order."""

from __future__ import annotations

from mldoctor.core.protocols import Diagnostic
from mldoctor.diagnostics.correlation import FeatureCorrelationDiagnostic
from mldoctor.diagnostics.data_integrity import DataIntegrityDiagnostic
from mldoctor.diagnostics.imbalance import ClassImbalanceDiagnostic
from mldoctor.diagnostics.leakage import LeakageDiagnostic
from mldoctor.diagnostics.outliers import OutlierDiagnostic
from mldoctor.diagnostics.overfitting import OverfittingDiagnostic
from mldoctor.diagnostics.prediction_errors import PredictionErrorDiagnostic


def builtin_diagnostics() -> list[Diagnostic]:
    return [
        DataIntegrityDiagnostic(),
        ClassImbalanceDiagnostic(),
        FeatureCorrelationDiagnostic(),
        OutlierDiagnostic(),
        LeakageDiagnostic(),
        OverfittingDiagnostic(),
        PredictionErrorDiagnostic(),
    ]
