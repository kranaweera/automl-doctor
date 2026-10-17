from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from mldoctor.statistics.correlations import pairwise_correlations, pearson_spearman


def _frame(seed: int = 0, n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n) * 1000 + 1e6
    return pd.DataFrame(
        {
            "a": a,
            "b": 2 * a,  # perfect linear
            "c": np.exp(rng.normal(size=n)),
            "d": a**3 + rng.normal(size=n),  # monotone, non-linear
        }
    )


def test_b_equals_two_a_is_perfect() -> None:
    pairs = {(p.a, p.b): p for p in pairwise_correlations(_frame())}
    ab = pairs[("a", "b")]
    assert ab.pearson == pytest.approx(1.0, abs=1e-12)
    assert ab.spearman == pytest.approx(1.0, abs=1e-12)
    assert ab.n == 300


def test_matches_scipy_with_missing_values() -> None:
    df = _frame(1)
    df.loc[:40, "c"] = np.nan
    df.loc[100:120, "d"] = np.nan
    df.loc[5, "a"] = np.inf  # non-finite counts as missing
    for pair in pairwise_correlations(df):
        joint = np.isfinite(df[pair.a]) & np.isfinite(df[pair.b])
        assert pair.n == int(joint.sum())
        expected = stats.pearsonr(df[pair.a][joint], df[pair.b][joint])[0]
        assert pair.pearson == pytest.approx(expected, abs=1e-9)
        assert pair.spearman == pytest.approx(
            stats.spearmanr(df[pair.a][joint], df[pair.b][joint])[0], abs=1e-9
        )
        assert pair.spearman_exact


def test_constant_column_is_undefined_not_zero() -> None:
    df = pd.DataFrame({"x": np.arange(20.0), "k": np.ones(20)})
    (pair,) = pairwise_correlations(df)
    assert pair.pearson is None and pair.spearman is None
    assert pair.undefined_reason == "constant_on_pair"


def test_constant_on_joint_rows_only() -> None:
    # k varies overall but is constant on the rows where x is observed.
    x = np.r_[np.arange(15.0), np.full(5, np.nan)]
    k = np.r_[np.ones(15), np.arange(5.0) + 2]
    df = pd.DataFrame({"x": x, "k": k})
    (pair,) = pairwise_correlations(df)
    assert pair.undefined_reason == "constant_on_pair"


def test_sample_size_floor() -> None:
    df = pd.DataFrame({"x": np.arange(8.0), "y": np.arange(8.0) * 3})
    (pair,) = pairwise_correlations(df, min_n=10)
    assert pair.undefined_reason == "insufficient_pairs"
    assert pair.pearson is None


@given(st.integers(0, 10_000))
@settings(max_examples=25, deadline=None)
def test_symmetry_and_row_permutation_invariance(seed: int) -> None:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(40, 3)), columns=["p", "q", "r"])
    df["q"] = df["q"] + df["p"]
    forward = {(p.a, p.b): p for p in pairwise_correlations(df)}
    reverse = {(p.b, p.a): p for p in pairwise_correlations(df[["r", "q", "p"]])}
    permuted = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    shuffled = {(p.a, p.b): p for p in pairwise_correlations(permuted)}
    for key, pair in forward.items():
        assert reverse[key].pearson == pytest.approx(pair.pearson, abs=1e-10)
        assert reverse[key].spearman == pytest.approx(pair.spearman, abs=1e-10)
        assert shuffled[key].pearson == pytest.approx(pair.pearson, abs=1e-10)
        assert shuffled[key].spearman == pytest.approx(pair.spearman, abs=1e-10)


def test_pearson_spearman_helper() -> None:
    n, r, rho = pearson_spearman(np.array([1.0, 2, 3, np.nan]), np.array([2.0, 4, 7, 1]))
    assert n == 3 and r is not None and rho == pytest.approx(1.0)
    assert pearson_spearman(np.ones(5), np.arange(5.0))[1] is None
