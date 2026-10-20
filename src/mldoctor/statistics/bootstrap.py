"""Nonparametric bootstrap for held-out metrics.

* **Row bootstrap**: resample the n evaluation rows with replacement (independent rows).
* **Cluster (group) bootstrap**: resample the G groups with replacement and include *all*
  member rows of every drawn group, preserving multiplicity (a group drawn twice
  contributes its rows twice). Use this whenever rows within a group are dependent, e.g.
  repeated measurements of a participant.

The interval is the percentile interval ``[q_{α/2}, q_{1−α/2}]`` of the valid replicate
statistics (linear interpolation). Replicates on which the statistic is undefined (e.g.
a resample missing a class) are dropped and counted; the interval is then conditional on
valid replicates, which can bias it, so a low valid fraction is surfaced to the caller.

Estimand: performance of the *fixed, already-trained* model on the population represented
by the evaluation set. The interval does not capture variability from re-training the model,
from choosing a different split, or from future (temporal) drift. Percentile intervals can
under-cover with small or imbalanced evaluation sets.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

Statistic = Callable[[NDArray[np.intp]], float | None]
Unit = Literal["row", "group"]


@dataclass(frozen=True)
class Resampler:
    """Pre-drawn bootstrap index sets, shared across metrics for coherence."""

    unit: Unit
    n_rows: int
    n_units: int
    replicates: list[NDArray[np.intp]]


def row_resampler(n_rows: int, n_boot: int, rng: np.random.Generator) -> Resampler:
    if n_rows < 1:
        raise ValueError("cannot bootstrap an empty evaluation set")
    draws = rng.integers(0, n_rows, size=(n_boot, n_rows))
    return Resampler("row", n_rows, n_rows, [draws[b].astype(np.intp) for b in range(n_boot)])


def group_resampler(
    groups: Sequence[object] | NDArray[np.object_], n_boot: int, rng: np.random.Generator
) -> Resampler:
    """Cluster bootstrap: draw G group labels with replacement; keep all member rows."""
    codes, uniques = pd.factorize(pd.Series(list(groups)), sort=True)
    n_groups = len(uniques)
    if n_groups < 1:
        raise ValueError("cannot bootstrap without groups")
    order = np.argsort(codes, kind="stable")
    boundaries = np.cumsum(np.bincount(codes, minlength=n_groups))
    members = np.split(order.astype(np.intp), boundaries[:-1])
    draws = rng.integers(0, n_groups, size=(n_boot, n_groups))
    replicates = [np.concatenate([members[g] for g in draws[b]]) for b in range(n_boot)]
    return Resampler("group", len(codes), n_groups, replicates)


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float | None
    estimate_reason: str | None
    ci_low: float | None
    ci_high: float | None
    ci_reason: str | None
    confidence: float
    n_replicates: int
    n_valid: int
    replicate_sd: float | None

    @property
    def valid_fraction(self) -> float:
        return self.n_valid / self.n_replicates if self.n_replicates else 0.0

    @property
    def width(self) -> float | None:
        if self.ci_low is None or self.ci_high is None:
            return None
        return self.ci_high - self.ci_low


def bootstrap_statistic(
    statistic: Statistic,
    resampler: Resampler,
    *,
    confidence: float = 0.95,
    full_sample_reason: str | None = None,
) -> BootstrapResult:
    """Percentile CI for ``statistic`` evaluated on each pre-drawn replicate."""
    all_rows = np.arange(resampler.n_rows, dtype=np.intp)
    estimate = statistic(all_rows)
    values = []
    for idx in resampler.replicates:
        value = statistic(idx)
        if value is not None and np.isfinite(value):
            values.append(value)
    n_valid = len(values)
    b = len(resampler.replicates)
    if estimate is None:
        return BootstrapResult(None, full_sample_reason or "metric undefined on the full "
                               "evaluation set", None, None, "point estimate undefined",
                               confidence, b, n_valid, None)
    if n_valid < 2:
        return BootstrapResult(float(estimate), None, None, None,
                               f"only {n_valid} valid bootstrap replicate(s)", confidence, b,
                               n_valid, None)
    arr = np.asarray(values, dtype=float)
    alpha = 1.0 - confidence
    low, high = np.quantile(arr, [alpha / 2, 1 - alpha / 2])
    return BootstrapResult(
        estimate=float(estimate),
        estimate_reason=None,
        ci_low=float(low),
        ci_high=float(high),
        ci_reason=None,
        confidence=confidence,
        n_replicates=b,
        n_valid=n_valid,
        replicate_sd=float(arr.std(ddof=1)),
    )


# ---------------------------------------------------------------------------------------
# Metric statistics on index sets
# ---------------------------------------------------------------------------------------


def regression_statistics(y: NDArray[np.float64], y_hat: NDArray[np.float64]) -> dict[
    str, Statistic
]:
    def mae(idx: NDArray[np.intp]) -> float | None:
        return float(np.mean(np.abs(y_hat[idx] - y[idx])))

    def rmse(idx: NDArray[np.intp]) -> float | None:
        return float(np.sqrt(np.mean((y_hat[idx] - y[idx]) ** 2)))

    def r2(idx: NDArray[np.intp]) -> float | None:
        yy = y[idx]
        if yy.size < 2:
            return None
        ss_tot = float(np.sum((yy - yy.mean()) ** 2))
        if ss_tot == 0:
            return None
        return 1.0 - float(np.sum((yy - y_hat[idx]) ** 2)) / ss_tot

    return {"mae": mae, "rmse": rmse, "r2": r2}


def classification_statistics(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> dict[str, Statistic]:
    """Accuracy, balanced accuracy and macro-F1 on encoded labels.

    A replicate is *invalid* for balanced accuracy and macro-F1 when it lacks any class that
    is present in the full evaluation set: the metric would then average over a different
    set of classes, i.e. estimate a different quantity. Accuracy is always defined.
    """
    index = {label: i for i, label in enumerate(labels)}
    t = np.array([index[v] for v in y_true], dtype=np.intp)
    p = np.array([index[v] for v in y_pred], dtype=np.intp)
    k = len(labels)
    present_full = np.unique(t)

    def confusion(idx: NDArray[np.intp]) -> NDArray[np.int64]:
        return np.bincount(t[idx] * k + p[idx], minlength=k * k).reshape(k, k)

    def accuracy(idx: NDArray[np.intp]) -> float | None:
        return float(np.mean(t[idx] == p[idx])) if idx.size else None

    def balanced_accuracy(idx: NDArray[np.intp]) -> float | None:
        if present_full.size < 2:
            return None
        cm = confusion(idx)
        support = cm.sum(axis=1)
        if np.any(support[present_full] == 0):
            return None
        return float(np.mean(np.diag(cm)[present_full] / support[present_full]))

    def macro_f1(idx: NDArray[np.intp]) -> float | None:
        cm = confusion(idx)
        support = cm.sum(axis=1)
        if np.any(support[present_full] == 0):
            return None
        tp = np.diag(cm)
        denom = 2 * tp + (cm.sum(axis=0) - tp) + (support - tp)
        used = denom > 0
        if not used.any():
            return None
        return float(np.mean(2 * tp[used] / denom[used]))

    return {"accuracy": accuracy, "balanced_accuracy": balanced_accuracy, "macro_f1": macro_f1}
