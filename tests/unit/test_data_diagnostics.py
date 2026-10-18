from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mldoctor.api import build_context
from mldoctor.core.config import DoctorConfig
from mldoctor.core.engine import run_one
from mldoctor.core.schemas import DiagnosticResult, Finding
from mldoctor.diagnostics.correlation import FeatureCorrelationDiagnostic
from mldoctor.diagnostics.data_integrity import DataIntegrityDiagnostic
from mldoctor.diagnostics.imbalance import ClassImbalanceDiagnostic
from mldoctor.diagnostics.outliers import OutlierDiagnostic
from tests.fixtures.synthetic import classification_frame, regression_frame


def _codes(result: DiagnosticResult) -> dict[str, list[Finding]]:
    out: dict[str, list[Finding]] = {}
    for f in result.findings:
        out.setdefault(f.code, []).append(f)
    return out


# --- data integrity ----------------------------------------------------------------------


def test_forty_percent_missing_is_high_on_correct_split() -> None:
    train = regression_frame(500, 1)
    test = regression_frame(200, 2, "t")
    test.loc[: int(0.4 * 200) - 1, "x2"] = np.nan  # exactly 40% missing in test only
    ctx, _ = build_context(train=train, test=test, target="y", task="regression",
                           row_id="row_id")
    result = run_one(DataIntegrityDiagnostic(), ctx)
    (finding,) = _codes(result)["HIGH_MISSINGNESS"]
    assert finding.severity == "high"
    assert finding.evidence["split"] == "test"
    assert finding.evidence["missing_fraction"] == {"x2": 0.4}
    assert result.status == "failed"
    assert result.metrics["splits"]["train"]["features_with_missing"] == 0


def test_moderate_missingness_is_medium() -> None:
    train = regression_frame(200)
    train.loc[:19, "x1"] = np.nan  # 10%
    ctx, _ = build_context(train=train, target="y", task="regression", row_id="row_id")
    (finding,) = _codes(run_one(DataIntegrityDiagnostic(), ctx))["HIGH_MISSINGNESS"]
    assert finding.severity == "medium"
    assert finding.evidence["missing_fraction"] == {"x1": 0.1}


def test_constant_and_infinite_values() -> None:
    train = regression_frame(100)
    train["k"] = 7.0
    train.loc[3, "x1"] = np.inf
    train.loc[4, "x1"] = -np.inf
    train.loc[5, "x1"] = np.nan
    ctx, _ = build_context(train=train, target="y", task="regression", row_id="row_id")
    codes = _codes(run_one(DataIntegrityDiagnostic(), ctx))
    assert codes["CONSTANT_FEATURE"][0].evidence["features"] == ["k"]
    inf = codes["INVALID_NUMERIC_VALUES"][0].evidence["features"]
    assert inf == {"x1": {"positive_infinity": 1, "negative_infinity": 1}}


def test_duplicates_with_discordant_targets_warn_not_fail() -> None:
    train = pd.DataFrame(
        {"a": [1, 1, 1, 2, 3, 3], "b": ["x", "x", "x", "y", "z", "z"], "y": [0, 1, 0, 1, 1, 1]}
    )
    ctx, _ = build_context(train=train, target="y", task="classification")
    result = run_one(DataIntegrityDiagnostic(), ctx)
    codes = _codes(result)
    dup = codes["DUPLICATE_ROWS"][0]
    assert dup.evidence["duplicate_feature_rows"] == 3  # 2 repeats of (1,x) + 1 of (3,z)
    assert dup.evidence["duplicate_full_rows"] == 2  # (1,x,0) once, (3,z,1) once
    disc = codes["DISCORDANT_DUPLICATES"][0]
    assert disc.severity == "medium"
    assert disc.evidence["discordant_groups"] == 1
    assert disc.evidence["rows_involved"] == 3
    assert result.status == "warning"


def test_mixed_types_and_dtype_mismatch() -> None:
    train = pd.DataFrame({"a": ["1", "2", "?", "4", "5"], "b": [1.0, 2, 3, 4, 5],
                          "y": [1.0, 2, 3, 4, 5]})
    test = pd.DataFrame({"a": ["1", "2", "3", "4", "5"], "b": ["x", "y", "z", "w", "v"],
                         "y": [1.0, 2, 3, 4, 5]})
    ctx, _ = build_context(train=train, test=test, target="y", task="regression")
    codes = _codes(run_one(DataIntegrityDiagnostic(), ctx))
    mixed = codes["MIXED_TYPES"][0]
    assert mixed.evidence["features"]["a"]["non_numeric_count"] == 1
    assert mixed.severity == "medium"
    mismatch = {m["feature"] for m in codes["DTYPE_MISMATCH"][0].evidence["features"]}
    assert mismatch == {"b"}  # "a" is text in both in-memory frames
    assert "b" not in ctx.numeric_features


def test_clean_data_passes() -> None:
    ctx, _ = build_context(train=regression_frame(300), target="y", task="regression",
                           row_id="row_id")
    result = run_one(DataIntegrityDiagnostic(), ctx)
    assert result.status == "passed"
    assert result.findings == []


# --- class imbalance ---------------------------------------------------------------------


def test_990_to_10_is_severe_imbalance() -> None:
    train = pd.DataFrame({"x": np.arange(1000.0), "y": [0] * 990 + [1] * 10})
    ctx, _ = build_context(train=train, target="y", task="classification")
    result = run_one(ClassImbalanceDiagnostic(), ctx)
    (finding,) = _codes(result)["CLASS_IMBALANCE"]
    assert finding.severity == "high"
    assert finding.evidence["minority_count"] == 10
    assert finding.evidence["minority_proportion"] == pytest.approx(0.01)
    assert finding.evidence["imbalance_ratio_present_classes"] == pytest.approx(99.0)
    p = np.array([0.99, 0.01])
    expected_h = float(-(p * np.log(p)).sum() / np.log(2))
    assert result.metrics["splits"]["train"]["normalized_entropy"] == pytest.approx(
        expected_h, abs=1e-6
    )


def test_balanced_classes_pass_with_entropy_one() -> None:
    train = pd.DataFrame({"x": np.arange(300.0), "y": ["a", "b", "c"] * 100})
    ctx, _ = build_context(train=train, target="y", task="classification")
    result = run_one(ClassImbalanceDiagnostic(), ctx)
    assert result.status == "passed"
    assert result.metrics["splits"]["train"]["normalized_entropy"] == pytest.approx(1.0)
    assert result.metrics["splits"]["train"]["imbalance_ratio"] == pytest.approx(1.0)


def test_one_class_in_test_is_flagged_without_crash() -> None:
    train = classification_frame(200)
    test = classification_frame(50, 3, "t")
    test["y"] = 0
    ctx, _ = build_context(train=train, test=test, target="y", task="classification",
                           row_id="row_id")
    result = run_one(ClassImbalanceDiagnostic(), ctx)
    codes = _codes(result)
    assert codes["ONE_CLASS_TARGET"][0].evidence["split"] == "test"
    test_metrics = result.metrics["splits"]["test"]
    assert test_metrics["imbalance_ratio"] is None
    assert "zero examples" in test_metrics["imbalance_ratio_undefined_reason"]
    assert test_metrics["normalized_entropy"] == 0.0


def test_imbalance_skipped_for_regression() -> None:
    ctx, _ = build_context(train=regression_frame(50), target="y", task="regression")
    result = run_one(ClassImbalanceDiagnostic(), ctx)
    assert result.status == "skipped"
    assert "regression" in (result.skip_reason or "")


def test_single_declared_class_entropy_undefined() -> None:
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": ["a", "a", "a"]})
    ctx, _ = build_context(train=train, target="y", task="classification")
    result = run_one(ClassImbalanceDiagnostic(), ctx)
    m = result.metrics["splits"]["train"]
    assert m["normalized_entropy"] is None
    assert "K=1" in m["normalized_entropy_undefined_reason"]
    assert result.status == "failed"


# --- correlation ---------------------------------------------------------------------------


def test_feature_b_twice_a_flagged() -> None:
    train = regression_frame(200)
    train["x1_double"] = 2 * train["x1"]
    ctx, _ = build_context(train=train, target="y", task="regression", row_id="row_id")
    result = run_one(FeatureCorrelationDiagnostic(), ctx)
    (finding,) = _codes(result)["HIGH_FEATURE_CORRELATION"]
    assert finding.severity == "medium"
    (pair,) = finding.evidence["pairs"]
    assert {pair["feature_a"], pair["feature_b"]} == {"x1", "x1_double"}
    assert pair["pearson"] == pytest.approx(1.0)
    assert "y" not in {p["feature_a"] for p in result.metrics["top_pairs"]}


def test_near_threshold_correlation_is_low() -> None:
    rng = np.random.default_rng(5)
    n = 2000
    a = rng.normal(size=n)
    b = 0.95 * a + np.sqrt(1 - 0.95**2) * rng.normal(size=n)
    train = pd.DataFrame({"a": a, "b": b, "y": rng.normal(size=n)})
    ctx, _ = build_context(train=train, target="y", task="regression")
    (finding,) = _codes(run_one(FeatureCorrelationDiagnostic(), ctx))["HIGH_FEATURE_CORRELATION"]
    assert finding.severity == "low"


def test_constant_feature_correlation_undefined() -> None:
    train = regression_frame(50)
    train["k"] = 1.0
    ctx, _ = build_context(train=train, target="y", task="regression", row_id="row_id")
    result = run_one(FeatureCorrelationDiagnostic(), ctx)
    assert result.metrics["constant_features"] == ["k"]
    assert result.metrics["pairs_undefined"]["constant_on_pair"] == 3


def test_correlation_skipped_with_single_numeric_feature() -> None:
    train = pd.DataFrame({"x": [1.0, 2, 3], "c": ["a", "b", "c"], "y": [1.0, 2, 3]})
    ctx, _ = build_context(train=train, target="y", task="regression")
    assert run_one(FeatureCorrelationDiagnostic(), ctx).status == "skipped"


def test_feature_cap_reported() -> None:
    train = regression_frame(50)
    cfg = DoctorConfig.model_validate({"correlation": {"max_features": 2}})
    ctx, _ = build_context(train=train, target="y", task="regression", row_id="row_id",
                           config=cfg)
    result = run_one(FeatureCorrelationDiagnostic(), ctx)
    assert result.metrics["features_analyzed"] == 2
    assert "features_beyond_cap" in result.skipped_subchecks


# --- outliers ------------------------------------------------------------------------------


def test_extreme_outlier_flagged_medium() -> None:
    train = regression_frame(300)
    train.loc[7, "x2"] = 60.0
    ctx, _ = build_context(train=train, target="y", task="regression", row_id="row_id")
    result = run_one(OutlierDiagnostic(), ctx)
    finding = _codes(result)["OUTLIER_CANDIDATES"][0]
    assert finding.severity == "medium"
    x2 = next(f for f in finding.evidence["features"] if f["feature"] == "x2")
    assert x2["extreme"] == 1
    assert result.status == "warning"


def test_outliers_zero_mad_explained() -> None:
    train = pd.DataFrame({"k": [5.0] * 95 + [6, 7, 8, 9, 1000], "y": np.arange(100.0)})
    ctx, _ = build_context(train=train, target="y", task="regression")
    result = run_one(OutlierDiagnostic(), ctx)
    deg = _codes(result)["DEGENERATE_SCALE"][0]
    assert deg.evidence["features"]["k"]["deviations_from_median_train"] == 5
    assert "OUTLIER_CANDIDATES" not in _codes(result)
    assert result.status == "passed"  # info only, no outlier claim


def test_test_outliers_use_training_reference() -> None:
    train = regression_frame(300)
    test = regression_frame(100, 9, "t")
    test.loc[0, "x3"] = 500.0
    ctx, _ = build_context(train=train, test=test, target="y", task="regression",
                           row_id="row_id")
    result = run_one(OutlierDiagnostic(), ctx)
    test_finding = next(
        f for f in result.findings
        if f.code == "OUTLIER_CANDIDATES" and f.evidence["split"] == "test"
    )
    features = test_finding.evidence["features"]
    assert any(f["feature"] == "x3" and f["extreme"] == 1 for f in features)
    assert result.metrics["reference_split"] == "train"
