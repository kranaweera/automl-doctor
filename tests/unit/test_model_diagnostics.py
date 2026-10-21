from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mldoctor.api import build_context
from mldoctor.core.config import DoctorConfig
from mldoctor.core.engine import run_one
from mldoctor.core.schemas import DiagnosticResult, Finding
from mldoctor.diagnostics.leakage import LeakageDiagnostic, binary_auc, canonical_row_keys
from mldoctor.diagnostics.overfitting import OverfittingDiagnostic
from mldoctor.diagnostics.prediction_errors import PredictionErrorDiagnostic
from tests.fixtures.synthetic import classification_frame, grouped_frame, regression_frame


def _codes(result: DiagnosticResult) -> dict[str, Finding]:
    return {f.code: f for f in result.findings}


def _preds(frame: pd.DataFrame, values: np.ndarray | list) -> pd.DataFrame:
    return pd.DataFrame({"row_id": frame["row_id"].to_numpy(), "y_pred": values})


# --- leakage -------------------------------------------------------------------------------


def _grouped_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = grouped_frame(n_groups=40, per_group=5)
    train = df[df["pid"] < "g030"]
    test = df[df["pid"] >= "g028"]  # g028, g029 shared
    test = test.assign(row_id=test["row_id"] + "_t", x=test["x"] + 1000)  # no row overlap
    return train, test


@pytest.mark.parametrize(
    ("goal", "severity"),
    [("unseen_groups", "critical"), ("unspecified", "medium"), ("unseen_rows", "info")],
)
def test_group_overlap_severity_depends_on_goal(goal: str, severity: str) -> None:
    train, test = _grouped_split()
    ctx, _ = build_context(train=train, test=test, target="y", task="regression",
                           row_id="row_id", group="pid",
                           config=DoctorConfig(evaluation_target=goal))  # type: ignore[arg-type]
    result = run_one(LeakageDiagnostic(), ctx)
    finding = _codes(result)["GROUP_OVERLAP"]
    assert finding.severity == severity
    assert finding.evidence["overlapping_groups"] == 2
    assert finding.evidence["test_rows_in_overlapping_groups"] == 10
    assert "g028" not in str(result.model_dump())  # identifiers never reported


def test_no_group_column_skips_group_check_explicitly() -> None:
    train, test = _grouped_split()
    ctx, _ = build_context(train=train, test=test, target="y", task="regression",
                           row_id="row_id")
    result = run_one(LeakageDiagnostic(), ctx)
    assert "group_overlap" in result.skipped_subchecks
    assert any("cannot be detected" in lim for lim in result.limitations)


def test_cross_split_duplicates_counted_both_sides() -> None:
    train = regression_frame(200, 1)
    test = regression_frame(100, 2, "t")
    test.iloc[:30, 1:4] = train.iloc[:30, 1:4].to_numpy()  # copy features of 30 train rows
    ctx, _ = build_context(train=train, test=test, target="y", task="regression",
                           row_id="row_id")
    result = run_one(LeakageDiagnostic(), ctx)
    finding = _codes(result)["CROSS_SPLIT_DUPLICATE"]
    assert finding.evidence["test_rows_in_train"] == 30
    assert finding.evidence["train_rows_in_test"] == 30
    assert finding.evidence["test_fraction"] == pytest.approx(0.3)
    assert finding.severity == "high"
    assert finding.evidence["test_rows_with_same_target"] == 0


def test_canonical_keys_preserve_missing_semantics() -> None:
    a = pd.DataFrame({"x": [1, np.nan, -0.0], "s": ["u", None, "v"]})
    b = pd.DataFrame({"x": [1.0, np.nan, 0.0], "s": ["u", None, "v"]})
    assert canonical_row_keys(a, ["x", "s"]) == canonical_row_keys(b, ["x", "s"])
    c = pd.DataFrame({"x": [np.nan], "s": ["NaN"]})
    d = pd.DataFrame({"x": [np.nan], "s": [None]})
    assert canonical_row_keys(c, ["x", "s"]) != canonical_row_keys(d, ["x", "s"])


def test_low_cardinality_overlap_downgraded() -> None:
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"a": rng.integers(0, 2, 300), "b": rng.integers(0, 2, 300),
                          "y": rng.normal(size=300)})
    test = pd.DataFrame({"a": rng.integers(0, 2, 50), "b": rng.integers(0, 2, 50),
                         "y": rng.normal(size=50)})
    ctx, _ = build_context(train=train, test=test, target="y", task="regression")
    finding = _codes(run_one(LeakageDiagnostic(), ctx))["CROSS_SPLIT_DUPLICATE"]
    assert finding.severity == "low"
    assert finding.evidence["low_cardinality_feature_space"] is True


def test_target_copies_detected() -> None:
    train = classification_frame(300)
    train["y_copy"] = train["y"]
    train["y_text"] = train["y"].map({0: "no", 1: "yes"})
    ctx, _ = build_context(train=train, target="y", task="classification", row_id="row_id")
    result = run_one(LeakageDiagnostic(), ctx)
    copies = {f["feature"]: f["kind"] for f in _codes(result)["POSSIBLE_TARGET_COPY"].evidence[
        "features"]}
    assert copies == {"y_copy": "exact_copy", "y_text": "one_to_one_recoding"}
    assert result.status == "failed"


def test_regression_linear_copy_and_association() -> None:
    train = regression_frame(300)
    rng = np.random.default_rng(1)
    train["scaled"] = 3 * train["y"] + 7
    train["noisy"] = train["y"] + rng.normal(scale=0.05, size=300)
    ctx, _ = build_context(train=train, target="y", task="regression", row_id="row_id")
    codes = _codes(run_one(LeakageDiagnostic(), ctx))
    copies = codes["POSSIBLE_TARGET_COPY"].evidence["features"]
    assert [c["feature"] for c in copies] == ["scaled"]
    assert copies[0]["kind"] == "exact_linear_transform"
    assoc = codes["SUSPICIOUS_TARGET_ASSOCIATION"]
    assert assoc.severity == "medium"  # screen only, never "failed"
    assert [a["feature"] for a in assoc.evidence["features"]] == ["noisy"]


def test_high_correlation_alone_is_not_called_leakage() -> None:
    # Strong but imperfect predictor (r ~ 0.9) must not be flagged.
    train = regression_frame(500)
    ctx, _ = build_context(train=train, target="y", task="regression", row_id="row_id")
    codes = _codes(run_one(LeakageDiagnostic(), ctx))
    assert "POSSIBLE_TARGET_COPY" not in codes
    assert "SUSPICIOUS_TARGET_ASSOCIATION" not in codes


def test_binary_auc_separation_and_name_screen() -> None:
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 400)
    train = pd.DataFrame({"score_after_outcome": y * 10 + rng.normal(size=400),
                          "x": rng.normal(size=400), "y": y})
    ctx, _ = build_context(train=train, target="y", task="classification")
    codes = _codes(run_one(LeakageDiagnostic(), ctx))
    assoc = codes["SUSPICIOUS_TARGET_ASSOCIATION"].evidence["features"]
    assert assoc[0]["measure"] == "auc" and assoc[0]["value"] > 0.99
    names = codes["SUSPICIOUS_FEATURE_NAME"].evidence["features"]
    assert names[0]["feature"] == "score_after_outcome"


def test_binary_auc_values() -> None:
    assert binary_auc(np.array([1.0, 2, 3, 4]), np.array([False, False, True, True])) == 1.0
    assert binary_auc(np.array([1.0, 1, 1, 1]), np.array([False, False, True, True])) == 0.5
    assert binary_auc(np.array([1.0, 2]), np.array([True, True])) is None


# --- prediction errors ------------------------------------------------------------------


def test_systematic_regression_bias() -> None:
    test = regression_frame(300, 4, "t")
    rng = np.random.default_rng(0)
    preds = _preds(test, test["y"] + 2.0 + rng.normal(scale=0.5, size=300))
    ctx, _ = build_context(train=regression_frame(100), test=test, predictions=preds,
                           target="y", task="regression", row_id="row_id")
    result = run_one(PredictionErrorDiagnostic(), ctx)
    bias = _codes(result)["SYSTEMATIC_BIAS"]
    assert bias.evidence["mean_signed_error"] == pytest.approx(2.0, abs=0.1)
    assert bias.evidence["bias_fraction"] == pytest.approx(4 / (4 + 0.25), abs=0.03)
    assert "over-predicts" in bias.message
    assert result.metrics["mae"] == pytest.approx(
        float(np.mean(np.abs(preds["y_pred"] - test["y"]))))


def test_unbiased_predictions_not_flagged() -> None:
    test = regression_frame(300, 4, "t")
    rng = np.random.default_rng(0)
    preds = _preds(test, test["y"] + rng.normal(scale=1.0, size=300))
    ctx, _ = build_context(train=regression_frame(100), test=test, predictions=preds,
                           target="y", task="regression", row_id="row_id")
    assert "SYSTEMATIC_BIAS" not in _codes(run_one(PredictionErrorDiagnostic(), ctx))


def test_constant_target_r2_undefined_surfaced() -> None:
    test = regression_frame(30, 4, "t")
    test["y"] = 5.0
    preds = _preds(test, np.linspace(4, 6, 30))
    ctx, _ = build_context(train=regression_frame(100), test=test, predictions=preds,
                           target="y", task="regression", row_id="row_id")
    result = run_one(PredictionErrorDiagnostic(), ctx)
    assert result.metrics["r2"] is None
    assert "constant" in result.metrics["undefined"]["r2"]
    assert _codes(result)["UNDEFINED_METRIC"].severity == "info"


def test_classification_collapse_and_never_predicted() -> None:
    test = classification_frame(200, 5, "t")
    preds = _preds(test, [0] * 200)
    ctx, _ = build_context(train=classification_frame(200), test=test, predictions=preds,
                           target="y", task="classification", row_id="row_id")
    result = run_one(PredictionErrorDiagnostic(), ctx)
    codes = _codes(result)
    assert codes["PREDICTION_COLLAPSE"].severity == "high"
    assert codes["CLASS_NEVER_PREDICTED"].evidence["classes"] == ["1"]
    per = result.metrics["per_class"]["1"]
    assert per["precision"] is None and per["undefined"]["precision"] == "class never predicted"
    assert per["recall"] == 0.0


def test_one_class_test_set_handled() -> None:
    test = classification_frame(50, 5, "t")
    test["y"] = 1
    preds = _preds(test, [1] * 45 + [0] * 5)
    ctx, _ = build_context(train=classification_frame(200), test=test, predictions=preds,
                           target="y", task="classification", row_id="row_id")
    result = run_one(PredictionErrorDiagnostic(), ctx)
    assert result.status != "error"
    assert result.metrics["balanced_accuracy"] is None
    assert result.metrics["accuracy"] == pytest.approx(0.9)
    assert "ONE_CLASS_TARGET" in _codes(result)


def test_shuffled_prediction_file_gives_identical_metrics(tmp_path) -> None:
    test = regression_frame(120, 6, "t")
    rng = np.random.default_rng(1)
    preds = _preds(test, test["y"] + rng.normal(size=120))
    shuffled = preds.sample(frac=1.0, random_state=3)
    p1, p2 = tmp_path / "p1.csv", tmp_path / "p2.csv"
    preds.to_csv(p1, index=False)
    shuffled.to_csv(p2, index=False)
    results = []
    for path in (p1, p2):
        ctx, _ = build_context(train=regression_frame(50), test=test, predictions=path,
                               target="y", task="regression", row_id="row_id")
        results.append(run_one(PredictionErrorDiagnostic(), ctx).metrics)
    assert results[0] == results[1]


# --- overfitting ----------------------------------------------------------------------


def _r2_preds(y: np.ndarray, r2: float, seed: int) -> np.ndarray:
    """Predictions whose R² against y is exactly ``r2`` (noise orthogonalised and scaled)."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=y.size)
    noise -= noise.mean()
    yc = y - y.mean()
    noise -= yc * (noise @ yc) / (yc @ yc)
    ss_tot = float(yc @ yc)
    noise *= np.sqrt((1 - r2) * ss_tot / float(noise @ noise))
    return y + noise


def _overfit_ctx(train_r2: float, test_r2: float):
    train = regression_frame(400, 1)
    test = regression_frame(200, 2, "t")
    return build_context(
        train=train, test=test,
        train_predictions=_preds(train, _r2_preds(train["y"].to_numpy(), train_r2, 0)),
        predictions=_preds(test, _r2_preds(test["y"].to_numpy(), test_r2, 1)),
        target="y", task="regression", row_id="row_id",
    )[0]


def test_r2_95_vs_20_flagged_high() -> None:
    result = run_one(OverfittingDiagnostic(), _overfit_ctx(0.95, 0.20))
    gap = result.metrics["gaps"]["r2"]
    assert gap["train"] == pytest.approx(0.95)
    assert gap["test"] == pytest.approx(0.20)
    assert gap["gap"] == pytest.approx(0.75)
    finding = _codes(result)["GENERALIZATION_GAP"]
    assert finding.severity == "high"
    assert result.status == "failed"
    mae = result.metrics["gaps"]["mae"]
    assert mae["direction"] == "lower_is_better"
    assert mae["gap"] == pytest.approx(mae["test"] - mae["train"])
    assert mae["gap"] > 0


def test_small_gap_passes() -> None:
    result = run_one(OverfittingDiagnostic(), _overfit_ctx(0.80, 0.78))
    assert "GENERALIZATION_GAP" not in _codes(result)
    assert result.status == "passed"


def test_test_better_than_train_is_info() -> None:
    result = run_one(OverfittingDiagnostic(), _overfit_ctx(0.50, 0.90))
    assert _codes(result)["TEST_OUTPERFORMS_TRAIN"].severity == "info"
    assert "GENERALIZATION_GAP" not in _codes(result)


def test_skipped_without_train_predictions() -> None:
    test = regression_frame(50, 2, "t")
    ctx, _ = build_context(train=regression_frame(100), test=test,
                           predictions=_preds(test, test["y"]), target="y",
                           task="regression", row_id="row_id")
    result = run_one(OverfittingDiagnostic(), ctx)
    assert result.status == "skipped"
    assert "train-predictions" in (result.skip_reason or "")


def test_gap_not_computed_when_metric_undefined() -> None:
    train = classification_frame(200, 1)
    test = classification_frame(60, 2, "t")
    test["y"] = 0
    ctx, _ = build_context(
        train=train, test=test, target="y", task="classification", row_id="row_id",
        train_predictions=_preds(train, train["y"]),
        predictions=_preds(test, [0] * 50 + [1] * 10),
    )
    result = run_one(OverfittingDiagnostic(), ctx)
    bal = result.metrics["gaps"]["balanced_accuracy"]
    assert bal["gap"] is None
    assert "test:" in bal["gap_undefined_reason"]
    assert "balanced_accuracy" in _codes(result)["UNDEFINED_METRIC"].evidence["metrics"]
    acc = result.metrics["gaps"]["accuracy"]
    assert acc["gap"] == pytest.approx(1.0 - 50 / 60)


def test_uneven_class_recall_flagged() -> None:
    test = pd.DataFrame({"row_id": [f"t{i}" for i in range(120)], "x": np.arange(120.0),
                         "y": [0] * 100 + [1] * 20})
    preds = _preds(test, [0] * 100 + [1] * 2 + [0] * 18)
    train = test.assign(row_id=lambda d: d["row_id"] + "_tr")
    ctx, _ = build_context(train=train, test=test, predictions=preds,
                           target="y", task="classification", row_id="row_id")
    finding = _codes(run_one(PredictionErrorDiagnostic(), ctx))["UNEVEN_CLASS_PERFORMANCE"]
    assert finding.evidence["classes"] == [{"class": "1", "recall": 0.1, "support": 20}]
    assert finding.evidence["best_recall"] == 1.0
