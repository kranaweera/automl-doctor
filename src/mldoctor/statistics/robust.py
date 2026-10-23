"""Robust location/scale estimates and outlier screening.

Modified z-score (Iglewicz & Hoaglin, 1993)::

    z_i = 0.6745 * (x_i - median) / MAD,   MAD = median(|x_i - median|)

0.6745 is the 0.75 quantile of the standard normal, so ``MAD / 0.6745`` estimates the
standard deviation under normality. ``|z| > 3.5`` is the conventional candidate rule.

Assumptions and limitations:

* Non-finite values are excluded before estimation (they are reported elsewhere).
* The rule is univariate; multivariate outliers are not detected.
* For heavily discrete or skewed variables MAD can be 0. We then fall back to Tukey fences
  ``[Q1 - k·IQR, Q3 + k·IQR]`` (quantiles use linear interpolation, Hyndman–Fan type 7).
  If IQR is also 0 the scale is *degenerate*: we count values that differ from the median
  but do not call them statistically valid outliers.
* Flagged values are candidates for review, never automatic deletions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

MAD_CONSISTENCY = 0.6745

ScreenMethod = Literal["modified_z", "iqr", "degenerate", "insufficient_data"]


@dataclass(frozen=True)
class RobustScale:
    n: int
    median: float
    mad: float
    q1: float
    q3: float

    @property
    def iqr(self) -> float:
        return self.q3 - self.q1


def finite_values(values: ArrayLike) -> NDArray[np.float64]:
    arr: NDArray[np.float64] = np.asarray(values, dtype=float).ravel()
    finite: NDArray[np.float64] = arr[np.isfinite(arr)]
    return finite


def mad(values: ArrayLike) -> float:
    """Raw (unscaled) median absolute deviation of finite values."""
    x = finite_values(values)
    if x.size == 0:
        raise ValueError("MAD is undefined for an empty sample")
    return float(np.median(np.abs(x - np.median(x))))


def robust_scale(values: ArrayLike) -> RobustScale:
    x = finite_values(values)
    if x.size == 0:
        raise ValueError("robust scale is undefined for an empty sample")
    median = float(np.median(x))
    q1, q3 = (float(q) for q in np.quantile(x, [0.25, 0.75]))
    return RobustScale(
        n=int(x.size), median=median, mad=float(np.median(np.abs(x - median))), q1=q1, q3=q3
    )


def modified_z(values: ArrayLike, median: float, mad_value: float) -> NDArray[np.float64]:
    """Modified z-scores relative to a *reference* median/MAD (e.g. from training data)."""
    if not mad_value > 0:
        raise ValueError("modified z-score requires MAD > 0")
    x = np.asarray(values, dtype=float)
    return MAD_CONSISTENCY * (x - median) / mad_value


def iqr_fences(q1: float, q3: float, k: float = 1.5) -> tuple[float, float]:
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


@dataclass(frozen=True)
class SplitScreen:
    n: int
    candidates: int
    extreme: int
    max_score: float | None  # max |z| (modified_z) or max distance beyond fence in IQRs

    @property
    def fraction(self) -> float | None:
        return self.candidates / self.n if self.n else None


@dataclass(frozen=True)
class OutlierScreen:
    method: ScreenMethod
    reference: RobustScale | None
    train: SplitScreen
    test: SplitScreen | None
    degenerate_deviations_train: int = 0
    degenerate_deviations_test: int = 0


def _screen(
    x: NDArray[np.float64],
    ref: RobustScale,
    method: ScreenMethod,
    z_threshold: float,
    iqr_k: float,
    extreme_z: float,
    far_out_k: float,
) -> SplitScreen:
    if x.size == 0:
        return SplitScreen(0, 0, 0, None)
    if method == "modified_z":
        score = np.abs(modified_z(x, ref.median, ref.mad))
        return SplitScreen(
            n=int(x.size),
            candidates=int(np.sum(score > z_threshold)),
            extreme=int(np.sum(score > extreme_z)),
            max_score=float(np.max(score)),
        )
    lo, hi = iqr_fences(ref.q1, ref.q3, iqr_k)
    iqr = ref.iqr
    beyond = np.maximum(ref.q1 - x, x - ref.q3) / iqr  # distance from the box in IQR units
    return SplitScreen(
        n=int(x.size),
        candidates=int(np.sum((x < lo) | (x > hi))),
        extreme=int(np.sum(beyond > far_out_k)),
        max_score=float(max(np.max(beyond), 0.0)),
    )


def screen_outliers(
    train_values: ArrayLike,
    test_values: ArrayLike | None = None,
    *,
    z_threshold: float = 3.5,
    iqr_k: float = 1.5,
    use_iqr_fallback: bool = True,
    min_n: int = 5,
    extreme_z: float = 10.0,
    far_out_k: float = 3.0,
) -> OutlierScreen:
    """Screen one numeric variable; reference statistics come from training data only."""
    x_train = finite_values(train_values)
    x_test = finite_values(test_values) if test_values is not None else None
    if x_train.size < min_n:
        empty = SplitScreen(int(x_train.size), 0, 0, None)
        test_empty = SplitScreen(int(x_test.size), 0, 0, None) if x_test is not None else None
        return OutlierScreen("insufficient_data", None, empty, test_empty)
    ref = robust_scale(x_train)
    method: ScreenMethod
    if ref.mad > 0:
        method = "modified_z"
    elif use_iqr_fallback and ref.iqr > 0:
        method = "iqr"
    else:
        dev_train = int(np.sum(x_train != ref.median))
        dev_test = int(np.sum(x_test != ref.median)) if x_test is not None else 0
        return OutlierScreen(
            "degenerate",
            ref,
            SplitScreen(int(x_train.size), 0, 0, None),
            SplitScreen(int(x_test.size), 0, 0, None) if x_test is not None else None,
            degenerate_deviations_train=dev_train,
            degenerate_deviations_test=dev_test,
        )
    args = (ref, method, z_threshold, iqr_k, extreme_z, far_out_k)
    return OutlierScreen(
        method,
        ref,
        _screen(x_train, *args),
        _screen(x_test, *args) if x_test is not None else None,
    )
