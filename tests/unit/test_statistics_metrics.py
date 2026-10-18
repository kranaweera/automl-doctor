from __future__ import annotations

import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn import metrics as skm

from mldoctor.statistics.metrics import (
    class_distribution,
    classification_metrics,
    metric_gap,
    r2_score,
    regression_metrics,
)

LABELS = ["a", "b", "c"]


@given(st.integers(0, 10_000), st.integers(5, 200))
@settings(max_examples=60, deadline=None)
def test_classification_metrics_match_sklearn(seed: int, n: int) -> None:
    rng = np.random.default_rng(seed)
    y = rng.choice(LABELS, n).tolist()
    p = rng.choice(LABELS, n).tolist()
    cm = classification_metrics(y, p, LABELS)
    assert cm.accuracy.value == pytest.approx(skm.accuracy_score(y, p))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert cm.macro_f1.value == pytest.approx(skm.f1_score(y, p, average="macro"))
        if len(set(y)) >= 2:
            assert cm.balanced_accuracy.value == pytest.approx(skm.balanced_accuracy_score(y, p))
    assert cm.confusion == skm.confusion_matrix(y, p, labels=LABELS).tolist()


def test_zero_division_is_explicitly_undefined() -> None:
    cm = classification_metrics(["a", "a", "b"], ["a", "a", "a"], LABELS)
    per = {c.label: c for c in cm.per_class}
    assert per["b"].precision.value is None
    assert per["b"].precision.reason == "class never predicted"
    assert per["b"].recall.value == 0.0
    assert per["c"].recall.value is None and per["c"].f1.value is None
    assert per["c"].recall.reason == "class absent from y_true"


def test_one_class_balanced_accuracy_undefined() -> None:
    cm = classification_metrics(["a"] * 5, ["a", "b", "a", "a", "a"], LABELS)
    assert cm.balanced_accuracy.value is None
    assert "at least two classes" in (cm.balanced_accuracy.reason or "")
    assert cm.accuracy.value == pytest.approx(0.8)
    # macro over {a, b}: F1_a = 8/9, F1_b = 0 (spurious prediction)
    assert cm.macro_f1.value == pytest.approx((8 / 9 + 0) / 2)


def test_regression_metrics_known_values_and_sklearn() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([2.0, 2.0, 2.0, 6.0])
    m = regression_metrics(y, p)
    assert m["mae"].value == pytest.approx(1.0)
    assert m["mse"].value == pytest.approx(1.5)
    assert m["rmse"].value == pytest.approx(np.sqrt(1.5))
    assert m["mean_signed_error"].value == pytest.approx(0.5)
    assert m["r2"].value == pytest.approx(skm.r2_score(y, p))


def test_r2_undefined_cases() -> None:
    assert r2_score(np.array([3.0, 3.0, 3.0]), np.array([1.0, 2.0, 3.0])).value is None
    assert "constant" in (r2_score(np.ones(3), np.ones(3)).reason or "")
    assert r2_score(np.array([1.0]), np.array([1.0])).value is None


def test_metric_gap_direction() -> None:
    assert metric_gap(0.95, 0.20, higher_is_better=True) == pytest.approx(0.75)
    assert metric_gap(1.0, 3.0, higher_is_better=False) == pytest.approx(2.0)
    assert metric_gap(3.0, 1.0, higher_is_better=False) == pytest.approx(-2.0)


def test_class_distribution_undefined_ratio() -> None:
    dist = class_distribution(["a", "a", None], ["a", "b"])
    assert dist.n == 2
    assert dist.imbalance_ratio is None
    assert dist.absent_labels == ["b"]
    assert dist.entropy_normalized == 0.0
