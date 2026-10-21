from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mldoctor.api import build_context, diagnose
from mldoctor.core.config import DoctorConfig
from mldoctor.core.engine import run_one
from mldoctor.diagnostics.uncertainty import UncertaintyDiagnostic
from mldoctor.statistics.bootstrap import (
    bootstrap_statistic,
    classification_statistics,
    group_resampler,
    regression_statistics,
    row_resampler,
)
from tests.fixtures.synthetic import classification_frame, grouped_frame, regression_frame


def test_group_resampler_draws_whole_groups_with_multiplicity() -> None:
    groups = np.array(["a", "a", "a", "b", "c", "c"])
    res = group_resampler(groups, n_boot=200, rng=np.random.default_rng(0))
    assert res.unit == "group" and res.n_units == 3 and res.n_rows == 6
    members = {"a": {0, 1, 2}, "b": {3}, "c": {4, 5}}
    saw_duplicate_group = False
    for idx in res.replicates:
        counts = pd.Series(idx).value_counts()
        for g, rows in members.items():
            # every member row of a group appears the same number of times (whole groups)
            assert len({int(counts.get(r, 0)) for r in rows}) == 1, g
        drawn = sum(int(counts.get(min(rows), 0)) for rows in members.values())
        assert drawn == 3  # exactly G groups drawn
        saw_duplicate_group |= any(int(counts.get(min(r), 0)) >= 2 for r in members.values())
    assert saw_duplicate_group


def test_row_resampler_shapes() -> None:
    res = row_resampler(10, 50, np.random.default_rng(1))
    assert len(res.replicates) == 50
    assert all(r.size == 10 and r.max() < 10 for r in res.replicates)


def test_mae_interval_close_to_normal_approximation() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    y = rng.normal(size=n)
    y_hat = y + rng.normal(size=n)
    stat = regression_statistics(y, y_hat)["mae"]
    res = bootstrap_statistic(stat, row_resampler(n, 2000, np.random.default_rng(1)))
    abs_err = np.abs(y_hat - y)
    se = abs_err.std(ddof=1) / np.sqrt(n)
    assert res.estimate == pytest.approx(abs_err.mean())
    assert res.ci_low == pytest.approx(abs_err.mean() - 1.96 * se, abs=0.25 * se + 1e-3)
    assert res.ci_high == pytest.approx(abs_err.mean() + 1.96 * se, abs=0.25 * se + 1e-3)
    assert res.n_valid == 2000


def test_undefined_replicates_are_counted_not_crashing() -> None:
    # 1 example of class "b": most resamples miss it -> balanced accuracy undefined there.
    y = ["a"] * 19 + ["b"]
    p = ["a"] * 18 + ["b", "b"]
    stats = classification_statistics(y, p, ["a", "b"])
    res = bootstrap_statistic(stats["balanced_accuracy"],
                              row_resampler(20, 1000, np.random.default_rng(2)))
    assert res.estimate is not None
    # P(class b drawn at least once) = 1 - (19/20)^20 ~ 0.64
    assert res.valid_fraction == pytest.approx(1 - (19 / 20) ** 20, abs=0.05)
    acc = bootstrap_statistic(stats["accuracy"], row_resampler(20, 1000,
                                                               np.random.default_rng(2)))
    assert acc.n_valid == 1000


def _grouped_regression(n_groups: int = 40, per_group: int = 10):
    """Errors share a strong per-group offset, so rows within a group are dependent."""
    df = grouped_frame(n_groups=n_groups, per_group=per_group, seed=3)
    rng = np.random.default_rng(4)
    offset = np.repeat(rng.normal(scale=2.0, size=n_groups), per_group)
    preds = pd.DataFrame({"row_id": df["row_id"],
                          "y_pred": df["y"] + offset + rng.normal(scale=0.2, size=len(df))})
    return df, preds


def test_group_bootstrap_wider_than_row_bootstrap_under_clustering() -> None:
    df, preds = _grouped_regression()
    train = regression_frame(50).rename(columns={"x1": "x"})[["row_id", "x", "y"]]
    train["pid"] = "train_only"
    common = dict(train=train, test=df, predictions=preds, target="y", task="regression",
                  row_id="row_id", config=DoctorConfig(seed=7))
    ctx_group, _ = build_context(group="pid", **common)  # type: ignore[arg-type]
    ctx_row, _ = build_context(**common)  # type: ignore[arg-type]
    g = run_one(UncertaintyDiagnostic(), ctx_group).metrics
    r = run_one(UncertaintyDiagnostic(), ctx_row).metrics
    assert g["resampling_unit"] == "group" and g["n_units"] == 40
    assert r["resampling_unit"] == "row" and r["n_units"] == 400
    # Design effect ~ 1 + (m-1)·ICC ≈ 10 here, so width ratio should be ~ sqrt(10) ≈ 3.
    ratio = g["intervals"]["mae"]["width"] / r["intervals"]["mae"]["width"]
    assert ratio > 2.0


def test_seeded_intervals_reproducible() -> None:
    df, preds = _grouped_regression()
    train = df.iloc[:50].assign(row_id=lambda d: d["row_id"] + "_tr", pid="tr")

    def run(seed: int) -> dict:
        report = diagnose(train=train, test=df, predictions=preds, target="y",
                          task="regression", row_id="row_id", group="pid",
                          config={"seed": seed, "uncertainty": {"n_bootstrap": 300}},
                          diagnostics=["uncertainty"])
        return report.diagnostics[0].metrics["intervals"]

    assert run(42) == run(42)
    assert run(42) != run(43)


def test_uncertainty_independent_of_other_diagnostics() -> None:
    df, preds = _grouped_regression()
    train = df.iloc[:50].assign(row_id=lambda d: d["row_id"] + "_tr")
    kwargs = dict(train=train, test=df, predictions=preds, target="y", task="regression",
                  row_id="row_id", config={"uncertainty": {"n_bootstrap": 200}})
    alone = diagnose(diagnostics=["uncertainty"], **kwargs)  # type: ignore[arg-type]
    full = diagnose(**kwargs)  # type: ignore[arg-type]
    m_alone = alone.diagnostics[0].metrics["intervals"]
    m_full = next(d for d in full.diagnostics if d.id == "uncertainty").metrics["intervals"]
    assert m_alone == m_full


def test_small_and_imbalanced_test_set_flags() -> None:
    train = classification_frame(200)
    test = classification_frame(20, 4, "t")
    test["y"] = [0] * 19 + [1]
    preds = pd.DataFrame({"row_id": test["row_id"], "y_pred": [0] * 18 + [1, 1]})
    ctx, _ = build_context(train=train, test=test, predictions=preds, target="y",
                           task="classification", row_id="row_id")
    result = run_one(UncertaintyDiagnostic(), ctx)
    codes = {f.code: f for f in result.findings}
    assert codes["SMALL_EVALUATION_SET"].evidence["n_units"] == 20
    assert "balanced_accuracy" in codes["INSUFFICIENT_VALID_REPLICATES"].evidence["metrics"]
    assert "WIDE_METRIC_INTERVAL" in codes


def test_one_class_test_set_interval_undefined_with_reason() -> None:
    train = classification_frame(200)
    test = classification_frame(60, 4, "t")
    test["y"] = 1
    preds = pd.DataFrame({"row_id": test["row_id"], "y_pred": [1] * 50 + [0] * 10})
    ctx, _ = build_context(train=train, test=test, predictions=preds, target="y",
                           task="classification", row_id="row_id")
    result = run_one(UncertaintyDiagnostic(), ctx)
    bal = result.metrics["intervals"]["balanced_accuracy"]
    assert bal["estimate"] is None and bal["ci_low"] is None
    assert bal["undefined_reason"]
    assert result.metrics["intervals"]["accuracy"]["estimate"] == pytest.approx(50 / 60)
