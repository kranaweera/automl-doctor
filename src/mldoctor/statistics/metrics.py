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
