from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mldoctor.statistics.robust import (
    MAD_CONSISTENCY,
    iqr_fences,
    mad,
    modified_z,
    robust_scale,
    screen_outliers,
)

finite_floats = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)


def test_mad_known_value() -> None:
    # median = 3, |x - 3| = [2, 1, 0, 1, 97] -> MAD = 1
    assert mad([1, 2, 3, 4, 100]) == 1.0
    assert robust_scale([1, 2, 3, 4, 100]).median == 3.0


def test_modified_z_known_value() -> None:
    z = modified_z([3.0, 100.0], median=3.0, mad_value=1.0)
    assert z[0] == 0.0
    assert z[1] == pytest.approx(MAD_CONSISTENCY * 97)


def test_modified_z_rejects_zero_mad() -> None:
    with pytest.raises(ValueError):
        modified_z([1.0], 1.0, 0.0)


def test_non_finite_values_excluded() -> None:
    assert mad([1, 2, 3, np.nan, np.inf]) == 1.0


@given(st.lists(finite_floats, min_size=1, max_size=50), st.floats(-100, 100))
@settings(max_examples=100, deadline=None)
def test_mad_scale_equivariance(values: list[float], c: float) -> None:
    x = np.asarray(values)
    assert mad(c * x) == pytest.approx(abs(c) * mad(x), rel=1e-9, abs=1e-6)


@given(st.lists(finite_floats, min_size=1, max_size=50), finite_floats)
@settings(max_examples=100, deadline=None)
def test_mad_translation_invariance(values: list[float], shift: float) -> None:
    x = np.asarray(values)
    assert mad(x + shift) == pytest.approx(mad(x), rel=1e-6, abs=1e-3)


@given(st.lists(finite_floats, min_size=5, max_size=40), st.integers(0, 1000))
@settings(max_examples=50, deadline=None)
def test_screen_permutation_invariant(values: list[float], seed: int) -> None:
    x = np.asarray(values)
    perm = np.random.default_rng(seed).permutation(x)
    a, b = screen_outliers(x), screen_outliers(perm)
    assert (a.method, a.train.candidates) == (b.method, b.train.candidates)


def test_extreme_outlier_detected() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    x[10] = 40.0
    screen = screen_outliers(x)
    assert screen.method == "modified_z"
    assert screen.train.candidates >= 1
    assert screen.train.extreme == 1
    assert screen.train.max_score is not None and screen.train.max_score > 10


def test_mad_zero_falls_back_to_iqr() -> None:
    # > 50% identical values -> MAD = 0, but quartiles differ.
    x = np.array([0.0] * 60 + [1.0] * 20 + [2.0] * 19 + [50.0])
    ref = robust_scale(x)
    assert ref.mad == 0.0 and ref.iqr > 0
    screen = screen_outliers(x)
    assert screen.method == "iqr"
    lo, hi = iqr_fences(ref.q1, ref.q3)
    assert screen.train.candidates == int(np.sum((x < lo) | (x > hi)))
    assert screen.train.candidates >= 1


def test_mad_and_iqr_zero_is_degenerate_without_division() -> None:
    x = np.array([5.0] * 95 + [6.0, 7.0, 8.0, 9.0, 1000.0])
    with np.errstate(all="raise"):
        screen = screen_outliers(x, np.array([5.0, 9.0]))
    assert screen.method == "degenerate"
    assert screen.train.candidates == 0  # no outlier claim
    assert screen.degenerate_deviations_train == 5
    assert screen.degenerate_deviations_test == 1


def test_test_split_scored_with_training_reference() -> None:
    train = np.arange(1.0, 101.0)  # median 50.5, MAD 25
    test = np.array([50.0, 300.0])
    screen = screen_outliers(train, test)
    assert screen.reference is not None and screen.reference.mad == 25.0
    assert screen.test is not None and screen.test.candidates == 1
    expected = MAD_CONSISTENCY * (300 - 50.5) / 25
    assert screen.test.max_score == pytest.approx(expected)


def test_insufficient_data() -> None:
    assert screen_outliers([1.0, 2.0], min_n=5).method == "insufficient_data"
