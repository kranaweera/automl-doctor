"""Generalization-gap diagnostic.

The same metric is computed on training predictions and held-out predictions, both joined
to labels by row ID. A gap is only computed when the metric is defined on **both** splits.

Direction-aware gap (positive = worse on held-out):

* higher-is-better (accuracy, balanced accuracy, macro-F1, R², recall): ``train − test``
* lower-is-better (MAE, RMSE): ``test − train``; also reported relative to the training
  loss, ``(test − train) / train`` (undefined when the training loss is 0).

Severity thresholds are metric-specific and configurable (no blanket cut-off):
score metrics 0.05/0.15, R² 0.10/0.25 (absolute), losses 25%/50% (relative) for
medium/high. A gap is described as a *generalization gap*: overfitting is one explanation,
alongside distribution shift between splits, split errors and leakage into training.
Without training predictions the check is skipped — it never reports "no overfitting".
"""

from __future__ import annotations

from typing import Any, ClassVar

from mldoctor.core.context import DiagnosticContext
from mldoctor.core.protocols import BaseDiagnostic, Requirement
from mldoctor.core.schemas import SEVERITY_ORDER, DiagnosticResult, Finding, Severity
from mldoctor.recommendations.rules import make_finding
from mldoctor.statistics.metrics import (
    HIGHER_IS_BETTER,
    MetricValue,
    classification_metrics,
    metric_gap,
    regression_metrics,
    sig,
)


class OverfittingDiagnostic(BaseDiagnostic):
    id = "overfitting"
    title = "Generalization gap"
    description = "Train vs held-out performance with direction-aware, metric-specific gaps."
    requires: ClassVar[frozenset[Requirement]] = frozenset(
        {"test_predictions", "train_predictions", "test_target"}
    )
    limitations: ClassVar[tuple[str, ...]] = (
        "Gaps are point estimates; held-out metric uncertainty is reported by the "
        "uncertainty diagnostic.",
        "A gap does not identify its cause (overfitting, shift, split error or leakage).",
    )

    def _metrics(self, ctx: DiagnosticContext, split: str) -> tuple[
        dict[str, MetricValue], dict[str, MetricValue]
    ]:
        frame = ctx.split(split)
        preds = ctx.predictions(split)
        assert frame is not None and preds is not None
        if ctx.task == "regression":
            values = regression_metrics(
                frame[ctx.target].to_numpy(dtype=float), preds.to_numpy(dtype=float)
            )
            return {k: values[k] for k in ("mae", "rmse", "r2")}, {}
        cm = classification_metrics(
            frame[ctx.target].astype(str).tolist(), preds.astype(str).tolist(),
            ctx.class_labels or [],
        )
        recalls = {c.label: c.recall for c in cm.per_class}
        return cm.summary(), recalls

    def _severity(self, ctx: DiagnosticContext, name: str, gap: float,
                  relative: float | None) -> tuple[Severity | None, dict[str, float]]:
        cfg = ctx.config.overfitting
        if name in ("mae", "rmse"):
            thresholds = {"medium": cfg.loss_relative_gap_medium,
                          "high": cfg.loss_relative_gap_high, "basis": 1.0}
            value = relative
        elif name == "r2":
            thresholds = {"medium": cfg.r2_gap_medium, "high": cfg.r2_gap_high, "basis": 0.0}
            value = gap
        else:
            thresholds = {"medium": cfg.score_gap_medium, "high": cfg.score_gap_high,
                          "basis": 0.0}
            value = gap
        if value is None:
            return None, thresholds
        if value >= thresholds["high"]:
            return "high", thresholds
        if value >= thresholds["medium"]:
            return "medium", thresholds
        return None, thresholds

    def run(self, context: DiagnosticContext) -> DiagnosticResult:
        train_summary, train_recall = self._metrics(context, "train")
        test_summary, test_recall = self._metrics(context, "test")
        rows: dict[str, dict[str, Any]] = {}
        flagged: list[tuple[Severity, str]] = []
        reverse: list[str] = []
        undefined: dict[str, str] = {}

        for name in train_summary:
            tr, te = train_summary[name], test_summary[name]
            higher = HIGHER_IS_BETTER[name]
            row: dict[str, Any] = {
                "train": tr.value, "test": te.value,
                "direction": "higher_is_better" if higher else "lower_is_better",
                "gap": None, "relative_gap": None,
            }
            if tr.value is None or te.value is None:
                reason = "; ".join(
                    f"{split}: {mv.reason}" for split, mv in (("train", tr), ("test", te))
                    if mv.value is None
                )
                row["gap_undefined_reason"] = reason
                undefined[name] = reason
                rows[name] = row
                continue
            gap = metric_gap(tr.value, te.value, higher)
            relative: float | None = None
            if not higher:
                if tr.value > 0:
                    relative = gap / tr.value
                else:
                    row["relative_gap_undefined_reason"] = "training loss is 0 (perfect fit)"
            row["gap"] = sig(gap)
            row["relative_gap"] = None if relative is None else sig(relative)
            severity, thresholds = self._severity(context, name, gap, relative)
            row["thresholds"] = {"medium": thresholds["medium"], "high": thresholds["high"],
                                 "applied_to": "relative_gap" if not higher else "gap"}
            if severity is not None:
                row["severity"] = severity
                flagged.append((severity, name))
            negative_value = (relative if not higher else gap)
            if negative_value is not None and -negative_value >= thresholds["medium"]:
                reverse.append(name)
            rows[name] = {k: (sig(v) if isinstance(v, float) else v) for k, v in row.items()}

        recall_gaps: dict[str, Any] = {}
        for label, tr in train_recall.items():
            te = test_recall[label]
            if tr.value is None or te.value is None:
                recall_gaps[label] = {"gap": None, "reason": tr.reason or te.reason}
            else:
                recall_gaps[label] = {"train": sig(tr.value), "test": sig(te.value),
                                      "gap": sig(tr.value - te.value)}

        findings: list[Finding] = []
        if flagged:
            worst: Severity = max((s for s, _ in flagged), key=lambda s: SEVERITY_ORDER[s])
            names = [n for _, n in flagged]
            findings.append(
                make_finding(
                    "GENERALIZATION_GAP",
                    worst,
                    "held-out performance is worse than training on "
                    + ", ".join(
                        f"{n} (train {rows[n]['train']:.4g} → test {rows[n]['test']:.4g})"
                        for n in names
                    ),
                    {"metrics": {n: rows[n] for n in names}},
                    limitations=[
                        "Possible causes include overfitting, distribution shift between splits, "
                        "split errors and leakage into training data."
                    ],
                )
            )
        if reverse:
            findings.append(
                make_finding(
                    "TEST_OUTPERFORMS_TRAIN",
                    "info",
                    "held-out performance is better than training on " + ", ".join(reverse),
                    {"metrics": {n: rows[n] for n in reverse}},
                )
            )
        if undefined:
            findings.append(
                make_finding(
                    "UNDEFINED_METRIC",
                    "info",
                    f"gap not computed for {len(undefined)} metric(s) because the metric is "
                    "undefined on at least one split",
                    {"metrics": undefined},
                )
            )
        metrics: dict[str, Any] = {
            "n_train": len(context.train),
            "n_test": 0 if context.test is None else len(context.test),
            "gaps": rows,
            "gap_definition": "positive = worse on held-out; higher-is-better: train − test; "
                              "lower-is-better: test − train",
        }
        if recall_gaps:
            metrics["per_class_recall"] = recall_gaps
        return self.result(findings, metrics)
