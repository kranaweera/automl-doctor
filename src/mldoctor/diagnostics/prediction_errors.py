"""Held-out prediction-error analysis.

Regression (``e = ŷ − y``): MAE, MSE, RMSE, median |e|, mean signed error, R² (if defined),
signed-error quantiles, errors by target quintile and the slope of the least-squares fit of
``y`` on ``ŷ``. ``SYSTEMATIC_BIAS`` fires when the squared mean error is a large share of the
MSE: ``bias_fraction = mean(e)² / mean(e²) ≥ 0.25`` (configurable; scale-free because
``mean(e²) = mean(e)² + var(e)``).

Classification (hard labels): confusion matrix (rows = truth, columns = prediction),
support, per-class precision/recall/F1 with explicit undefined values, accuracy, balanced
accuracy and macro-F1 (policy in :func:`mldoctor.statistics.metrics.classification_metrics`).
Flags: a single predicted class (``PREDICTION_COLLAPSE``), classes never predicted, and
predicted-vs-true prevalence differences ≥ 0.15 (``SYSTEMATIC_BIAS``).

Patterns are described, not explained causally. No probability or calibration claims are
made because v0.1 accepts hard predictions only.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from mldoctor.core.context import DiagnosticContext
from mldoctor.core.protocols import BaseDiagnostic, Requirement
from mldoctor.core.schemas import DiagnosticResult, Finding
from mldoctor.recommendations.rules import make_finding
from mldoctor.statistics.metrics import (
    ClassificationMetrics,
    MetricValue,
    classification_metrics,
    regression_metrics,
    sig,
)

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def _mv(value: MetricValue) -> float | None:
    return None if value.value is None else sig(value.value)


def undefined_map(values: dict[str, MetricValue]) -> dict[str, str]:
    return {k: v.reason for k, v in values.items() if v.value is None and v.reason}


def classification_payload(cm: ClassificationMetrics) -> dict[str, Any]:
    return {
        "n": cm.n,
        "labels": cm.labels,
        "confusion_matrix": {"rows": "true", "columns": "predicted", "matrix": cm.confusion},
        "accuracy": _mv(cm.accuracy),
        "balanced_accuracy": _mv(cm.balanced_accuracy),
        "macro_f1": _mv(cm.macro_f1),
        "per_class": {
            c.label: {
                "support": c.support,
                "predicted": c.predicted,
                "precision": _mv(c.precision),
                "recall": _mv(c.recall),
                "f1": _mv(c.f1),
                "undefined": {
                    name: mv.reason
                    for name, mv in (("precision", c.precision), ("recall", c.recall),
                                     ("f1", c.f1))
                    if mv.value is None and mv.reason
                },
            }
            for c in cm.per_class
        },
        "undefined": undefined_map(cm.summary()),
    }


class PredictionErrorDiagnostic(BaseDiagnostic):
    id = "prediction_errors"
    title = "Prediction error analysis"
    description = "Residual summaries (regression) or confusion-matrix metrics (classification)."
    requires: ClassVar[frozenset[Requirement]] = frozenset({"test_predictions", "test_target"})
    limitations: ClassVar[tuple[str, ...]] = (
        "Point estimates on one evaluation set; see the uncertainty diagnostic for intervals.",
        "Error patterns are descriptive and do not identify causes.",
    )

    def run(self, context: DiagnosticContext) -> DiagnosticResult:
        if context.task == "regression":
            return self._regression(context)
        return self._classification(context)

    def _regression(self, ctx: DiagnosticContext) -> DiagnosticResult:
        assert ctx.test is not None and ctx.test_predictions is not None
        cfg = ctx.config.prediction_errors
        y = ctx.test[ctx.target].to_numpy(dtype=float)
        y_hat = ctx.test_predictions.to_numpy(dtype=float)
        values = regression_metrics(y, y_hat)
        error = y_hat - y
        n = int(y.size)
        findings: list[Finding] = []
        mse = float(np.mean(error**2))
        mean_error = float(np.mean(error))
        bias_fraction = mean_error**2 / mse if mse > 0 else None

        slope: float | None = None
        slope_reason: str | None = None
        if np.ptp(y_hat) == 0:
            slope_reason = "predictions are constant"
        else:
            slope = float(np.polyfit(y_hat, y, 1)[0])

        by_quintile: list[dict[str, Any]] = []
        if n >= 25 and np.ptp(y) > 0:
            edges = np.quantile(y, np.linspace(0, 1, 6))
            bins = np.clip(np.searchsorted(edges, y, side="right") - 1, 0, 4)
            for b in range(5):
                sel = bins == b
                if sel.any():
                    by_quintile.append(
                        {"quintile": b + 1, "n": int(sel.sum()),
                         "mean_signed_error": sig(float(error[sel].mean())),
                         "mae": sig(float(np.abs(error[sel]).mean()))}
                    )

        metrics: dict[str, Any] = {
            "n": n,
            "error_definition": "ŷ − y",
            **{k: _mv(v) for k, v in values.items()},
            "error_sd": sig(float(np.std(error, ddof=1))) if n > 1 else None,
            "bias_fraction": None if bias_fraction is None else sig(bias_fraction),
            "signed_error_quantiles": {
                f"q{int(q * 100):02d}": sig(float(np.quantile(error, q))) for q in QUANTILES
            },
            "slope_y_on_prediction": None if slope is None else sig(slope),
            "slope_undefined_reason": slope_reason,
            "error_by_target_quintile": by_quintile,
            "undefined": undefined_map(values),
        }

        if np.ptp(y_hat) == 0 and np.ptp(y) > 0:
            findings.append(
                make_finding(
                    "PREDICTION_COLLAPSE",
                    "high",
                    f"all {n} held-out predictions are identical while the target varies",
                    {"n": n},
                )
            )
        if (
            bias_fraction is not None
            and n >= cfg.min_n
            and bias_fraction >= cfg.bias_fraction_threshold
        ):
            direction = "over-predicts" if mean_error > 0 else "under-predicts"
            findings.append(
                make_finding(
                    "SYSTEMATIC_BIAS",
                    "medium",
                    f"model {direction} on average: mean signed error {mean_error:.4g} "
                    f"accounts for {bias_fraction:.0%} of the MSE",
                    {"mean_signed_error": sig(mean_error),
                     "bias_fraction": sig(bias_fraction),
                     "threshold": cfg.bias_fraction_threshold, "n": n},
                    limitations=["Constant offset only; error patterns that cancel on average "
                                 "(e.g. by target range) are shown in error_by_target_quintile."],
                )
            )
        if values["r2"].value is None:
            findings.append(
                make_finding(
                    "UNDEFINED_METRIC",
                    "info",
                    f"R² is undefined: {values['r2'].reason}",
                    {"metric": "r2", "reason": values["r2"].reason},
                )
            )
        return self.result(findings, metrics)

    def _classification(self, ctx: DiagnosticContext) -> DiagnosticResult:
        assert ctx.test is not None and ctx.test_predictions is not None
        cfg = ctx.config.prediction_errors
        labels = ctx.class_labels or []
        y = ctx.test[ctx.target].astype(str).tolist()
        y_hat = ctx.test_predictions.astype(str).tolist()
        cm = classification_metrics(y, y_hat, labels)
        metrics = classification_payload(cm)
        findings: list[Finding] = []

        matrix = np.asarray(cm.confusion)
        off: list[dict[str, Any]] = [
            {"true": labels[i], "predicted": labels[j], "count": int(matrix[i, j])}
            for i in range(len(labels))
            for j in range(len(labels))
            if i != j and matrix[i, j] > 0
        ]
        off.sort(key=lambda d: (-d["count"], d["true"], d["predicted"]))
        metrics["top_confusions"] = off[: cfg.top_confusions]

        true_prev = matrix.sum(axis=1) / cm.n
        pred_prev = matrix.sum(axis=0) / cm.n
        diff = pred_prev - true_prev
        metrics["prevalence"] = {
            label: {"true": sig(float(t)), "predicted": sig(float(p))}
            for label, t, p in zip(labels, true_prev, pred_prev, strict=True)
        }

        present = [c for c in cm.per_class if c.support > 0]
        predicted_classes = [c for c in cm.per_class if c.predicted > 0]
        if len(predicted_classes) == 1 and len(present) >= 2:
            findings.append(
                make_finding(
                    "PREDICTION_COLLAPSE",
                    "high",
                    f"every held-out row is predicted as '{predicted_classes[0].label}' although "
                    f"{len(present)} classes are present",
                    {"predicted_class": predicted_classes[0].label, "n": cm.n},
                )
            )
        never = [c.label for c in present if c.predicted == 0]
        if never:
            findings.append(
                make_finding(
                    "CLASS_NEVER_PREDICTED",
                    "medium",
                    f"{len(never)} class(es) present in the held-out data are never predicted",
                    {"classes": never,
                     "support": {c.label: c.support for c in present if c.predicted == 0}},
                )
            )
        worst = int(np.argmax(np.abs(diff))) if len(labels) else 0
        if len(labels) and abs(diff[worst]) >= cfg.prevalence_shift_threshold:
            direction = "over-predicted" if diff[worst] > 0 else "under-predicted"
            findings.append(
                make_finding(
                    "SYSTEMATIC_BIAS",
                    "medium",
                    f"class '{labels[worst]}' is {direction}: predicted share "
                    f"{pred_prev[worst]:.1%} vs true share {true_prev[worst]:.1%}",
                    {"class": labels[worst],
                     "true_prevalence": sig(float(true_prev[worst])),
                     "predicted_prevalence": sig(float(pred_prev[worst])),
                     "threshold": cfg.prevalence_shift_threshold},
                )
            )
        undefined = metrics["undefined"]
        if undefined:
            findings.append(
                make_finding(
                    "UNDEFINED_METRIC",
                    "info",
                    f"{len(undefined)} summary metric(s) are undefined on this evaluation set",
                    {"metrics": undefined},
                )
            )
        if len(present) < 2:
            findings.append(
                make_finding(
                    "ONE_CLASS_TARGET",
                    "medium",
                    f"the held-out set contains {len(present)} class(es); per-class and "
                    "balanced metrics are undefined or degenerate",
                    {"classes_present": [c.label for c in present]},
                )
            )
        return self.result(findings, metrics)
