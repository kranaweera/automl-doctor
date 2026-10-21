"""Distributional summaries and evaluation metrics.

Every function returns ``None`` (never NaN) for undefined quantities together with a
machine-readable reason, so callers can surface *why* a number is unavailable.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ClassDistribution:
    labels: list[str]
    counts: list[int]
    n: int
    entropy_normalized: float | None
    entropy_undefined_reason: str | None
    majority_count: int
    minority_count: int
    imbalance_ratio: float | None
    imbalance_ratio_undefined_reason: str | None
    absent_labels: list[str] = field(default_factory=list)

    @property
    def proportions(self) -> list[float]:
        return [c / self.n for c in self.counts] if self.n else [0.0 for _ in self.counts]

    @property
    def minority_proportion(self) -> float | None:
        return self.minority_count / self.n if self.n else None

    @property
    def observed_classes(self) -> int:
        return sum(1 for c in self.counts if c > 0)


def class_distribution(values: Sequence[str | None], labels: Sequence[str]) -> ClassDistribution:
    """Class counts over the declared ``labels`` (missing values are excluded).

    Normalized Shannon entropy ``H = -Σ p_k ln p_k / ln K`` uses ``K = len(labels)`` (the
    declared classes), so a class absent from this split lowers H. ``0 · ln 0 := 0``.
    H is undefined for K = 1 (``ln 1 = 0``) and for an empty sample.

    The majority/minority ratio uses the smallest *declared* class; when a declared class
    has zero count the ratio is undefined (infinite) and reported as ``None``.
    """
    present = [v for v in values if v is not None]
    tally = Counter(present)
    counts = [tally.get(label, 0) for label in labels]
    n = len(present)
    k = len(labels)
    entropy: float | None = None
    entropy_reason: str | None = None
    if n == 0:
        entropy_reason = "no labelled rows"
    elif k <= 1:
        entropy_reason = "only one class declared (K=1): normalized entropy is undefined"
    else:
        h = -sum((c / n) * math.log(c / n) for c in counts if c > 0)
        entropy = h / math.log(k)
    majority = max(counts) if counts else 0
    minority = min(counts) if counts else 0
    ratio: float | None = None
    ratio_reason: str | None = None
    if n == 0:
        ratio_reason = "no labelled rows"
    elif minority == 0:
        ratio_reason = "at least one declared class has zero examples"
    elif k <= 1:
        ratio_reason = "only one class declared"
    else:
        ratio = majority / minority
    return ClassDistribution(
        labels=list(labels),
        counts=counts,
        n=n,
        entropy_normalized=entropy,
        entropy_undefined_reason=entropy_reason,
        majority_count=majority,
        minority_count=minority,
        imbalance_ratio=ratio,
        imbalance_ratio_undefined_reason=ratio_reason,
        absent_labels=[label for label, c in zip(labels, counts, strict=True) if c == 0],
    )


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0 or not np.isfinite(denominator) or not np.isfinite(numerator):
        return None
    return float(numerator / denominator)


# ---------------------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------------------

HIGHER_IS_BETTER: dict[str, bool] = {
    "mae": False,
    "rmse": False,
    "mse": False,
    "median_absolute_error": False,
    "r2": True,
    "accuracy": True,
    "balanced_accuracy": True,
    "macro_f1": True,
    "recall": True,
}


@dataclass(frozen=True)
class MetricValue:
    """A metric value, or ``None`` with a reason when mathematically undefined."""

    value: float | None
    reason: str | None = None

    @property
    def defined(self) -> bool:
        return self.value is not None


def _undefined(reason: str) -> MetricValue:
    return MetricValue(None, reason)


def r2_score(y: np.ndarray, y_hat: np.ndarray) -> MetricValue:
    """Coefficient of determination ``1 - SS_res / SS_tot``.

    Undefined for fewer than two observations or a constant target (``SS_tot = 0``); we do
    not substitute 0 or 1 as some libraries do.
    """
    n = y.size
    if n < 2:
        return _undefined("R² requires at least 2 observations")
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0:
        return _undefined("R² is undefined when the target is constant (SS_tot = 0)")
    ss_res = float(np.sum((y - y_hat) ** 2))
    return MetricValue(1.0 - ss_res / ss_tot)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, MetricValue]:
    """MAE, MSE, RMSE, median absolute error, mean signed error ``mean(ŷ − y)`` and R²."""
    y = np.asarray(y_true, dtype=float)
    y_hat = np.asarray(y_pred, dtype=float)
    if y.shape != y_hat.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if y.size == 0:
        reason = "no observations"
        return {k: _undefined(reason) for k in (
            "mae", "mse", "rmse", "median_absolute_error", "mean_signed_error", "r2")}
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(y_hat))):
        raise ValueError("regression metrics require finite inputs (validate inputs first)")
    error = y_hat - y
    mse = float(np.mean(error**2))
    return {
        "mae": MetricValue(float(np.mean(np.abs(error)))),
        "mse": MetricValue(mse),
        "rmse": MetricValue(float(np.sqrt(mse))),
        "median_absolute_error": MetricValue(float(np.median(np.abs(error)))),
        "mean_signed_error": MetricValue(float(np.mean(error))),
        "r2": r2_score(y, y_hat),
    }


@dataclass(frozen=True)
class ClassMetrics:
    label: str
    support: int
    predicted: int
    tp: int
    precision: MetricValue
    recall: MetricValue
    f1: MetricValue


@dataclass(frozen=True)
class ClassificationMetrics:
    labels: list[str]
    confusion: list[list[int]]  # rows = true label, columns = predicted label
    n: int
    per_class: list[ClassMetrics]
    accuracy: MetricValue
    balanced_accuracy: MetricValue
    macro_f1: MetricValue

    def summary(self) -> dict[str, MetricValue]:
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_f1": self.macro_f1,
        }


def classification_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> ClassificationMetrics:
    """Confusion matrix and label-based metrics with an explicit zero-division policy.

    * precision_k = TP/(TP+FP); **undefined** if class k is never predicted.
    * recall_k    = TP/(TP+FN); **undefined** if class k has no true examples.
    * F1_k        = 2TP/(2TP+FP+FN); undefined only if k is absent from both y and ŷ.
    * accuracy    = mean(y == ŷ).
    * balanced accuracy = mean recall over classes present in ``y_true``; undefined unless
      at least two classes are present (with one class it collapses to a single recall).
    * macro-F1 = unweighted mean of F1_k over classes present in ``y_true`` or ``y_pred``
      (scikit-learn's default label set), so spurious predicted classes count as F1 = 0.
    """
    yt = list(y_true)
    yp = list(y_pred)
    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred must have the same length")
    index = {label: i for i, label in enumerate(labels)}
    unknown = {v for v in (*yt, *yp) if v not in index}
    if unknown:
        raise ValueError(f"labels not declared: {sorted(unknown)[:10]}")
    k = len(labels)
    cm = np.zeros((k, k), dtype=np.int64)
    if yt:
        np.add.at(cm, ([index[v] for v in yt], [index[v] for v in yp]), 1)
    n = int(cm.sum())
    per_class: list[ClassMetrics] = []
    for i, label in enumerate(labels):
        tp = int(cm[i, i])
        support = int(cm[i, :].sum())
        predicted = int(cm[:, i].sum())
        fp, fn = predicted - tp, support - tp
        precision = (
            MetricValue(tp / predicted) if predicted else _undefined("class never predicted")
        )
        recall = MetricValue(tp / support) if support else _undefined("class absent from y_true")
        denom = 2 * tp + fp + fn
        f1 = MetricValue(2 * tp / denom) if denom else _undefined("class absent from y and ŷ")
        per_class.append(ClassMetrics(label, support, predicted, tp, precision, recall, f1))

    if n == 0:
        reason = "no observations"
        return ClassificationMetrics(
            list(labels), cm.tolist(), 0, per_class,
            _undefined(reason), _undefined(reason), _undefined(reason),
        )
    accuracy = MetricValue(float(np.trace(cm)) / n)
    recalls = [c.recall.value for c in per_class if c.support > 0 and c.recall.value is not None]
    if len(recalls) < 2:
        balanced = _undefined(
            "balanced accuracy requires at least two classes in y_true "
            f"({len(recalls)} present)"
        )
    else:
        balanced = MetricValue(float(np.mean(recalls)))
    f1s = [c.f1.value for c in per_class if c.f1.value is not None]
    macro_f1 = MetricValue(float(np.mean(f1s))) if f1s else _undefined("no classes observed")
    return ClassificationMetrics(
        list(labels), cm.tolist(), n, per_class, accuracy, balanced, macro_f1
    )


def metric_gap(train: float, test: float, higher_is_better: bool) -> float:
    """Directional generalization gap; positive means worse on held-out data.

    higher-is-better: ``train − test``; lower-is-better (losses): ``test − train``.
    """
    return train - test if higher_is_better else test - train


def sig(value: float, digits: int = 10) -> float:
    """Round to ``digits`` significant digits (scale-independent, unlike ``round(x, k)``)."""
    if value == 0 or not math.isfinite(value):
        return float(value)
    return float(f"{value:.{digits}g}")


def fmt_number(value: float) -> str:
    """Human-readable number for messages (evidence keeps full precision)."""
    if value != 0 and (abs(value) >= 1e5 or abs(value) < 1e-3):
        return f"{value:.3g}"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.4g}"


def name_list(names: list[str], limit: int = 3) -> str:
    shown = ", ".join(f"'{n}'" for n in names[:limit])
    return shown + (f" and {len(names) - limit} more" if len(names) > limit else "")
