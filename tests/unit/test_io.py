from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mldoctor.api import build_context
from mldoctor.io.csv_loader import load_csv
from mldoctor.io.prediction_loader import align_predictions
from mldoctor.io.validation import InputValidationError, canonical_labels


def _eval_frame() -> pd.DataFrame:
    return pd.DataFrame({"row_id": ["a", "b", "c", "d"], "x": [1.0, 2, 3, 4], "y": [1, 0, 1, 1]})


def test_canonical_labels_unify_numeric_representations() -> None:
    assert canonical_labels(pd.Series([1, 0, 1])).tolist() == ["1", "0", "1"]
    assert canonical_labels(pd.Series([1.0, 0.0, np.nan])).tolist() == ["1", "0", None]
    assert canonical_labels(pd.Series(["1.0", "0", "cat"])).tolist() == ["1", "0", "cat"]
    assert canonical_labels(pd.Series([0.5, 1.0])).tolist() == ["0.5", "1.0"]
    assert canonical_labels(pd.Series([True, False])).tolist() == ["True", "False"]


def test_shuffled_predictions_join_by_id() -> None:
    ev = _eval_frame()
    preds = pd.DataFrame({"row_id": ["d", "b", "a", "c"], "y_pred": [0, 0, 1, 1]})
    aligned = align_predictions(preds, ev, "row_id", "classification", "p", ["0", "1"])
    # a->1, b->0, c->1, d->0 regardless of file order
    assert aligned.tolist() == ["1", "0", "1", "0"]
    assert aligned.index.equals(ev.index)


@pytest.mark.parametrize(
    ("ids", "message"),
    [
        (["a", "b", "c"], "without a prediction"),
        (["a", "b", "c", "d", "e"], "unknown row IDs"),
        (["a", "b", "c", "c"], "unique"),
    ],
)
def test_prediction_id_mismatches_raise(ids: list[str], message: str) -> None:
    preds = pd.DataFrame({"row_id": ids, "y_pred": [1.0] * len(ids)})
    with pytest.raises(InputValidationError, match=message):
        align_predictions(preds, _eval_frame(), "row_id", "regression", "p")


def test_nonfinite_regression_predictions_raise() -> None:
    preds = pd.DataFrame({"row_id": ["a", "b", "c", "d"], "y_pred": [1.0, np.inf, 0.0, 1.0]})
    with pytest.raises(InputValidationError, match="non-finite"):
        align_predictions(preds, _eval_frame(), "row_id", "regression", "p")


def test_unknown_class_label_raises() -> None:
    preds = pd.DataFrame({"row_id": ["a", "b", "c", "d"], "y_pred": [1, 0, 2, 1]})
    with pytest.raises(InputValidationError, match="not among the declared class labels"):
        align_predictions(preds, _eval_frame(), "row_id", "classification", "p", ["0", "1"])


def test_missing_prediction_column_raises() -> None:
    preds = pd.DataFrame({"row_id": ["a", "b", "c", "d"], "prediction": [1, 0, 1, 1]})
    with pytest.raises(InputValidationError, match="y_pred"):
        align_predictions(preds, _eval_frame(), "row_id", "regression", "p")


def test_csv_loader_rejects_bad_files(tmp_path: Path) -> None:
    with pytest.raises(InputValidationError, match="not found"):
        load_csv(tmp_path / "missing.csv", "train")
    with pytest.raises(InputValidationError, match="only local files"):
        load_csv("https://example.com/data.csv", "train")
    dup = tmp_path / "dup.csv"
    dup.write_text("a,a,b\n1,2,3\n")
    with pytest.raises(InputValidationError, match="duplicate column"):
        load_csv(dup, "train")
    empty = tmp_path / "empty.csv"
    empty.write_text("a,b\n")
    with pytest.raises(InputValidationError, match="no data rows"):
        load_csv(empty, "train")


def test_ids_are_read_as_strings(tmp_path: Path) -> None:
    path = tmp_path / "t.csv"
    path.write_text("row_id,x\n007,1\n8,2\n")
    table = load_csv(path, "train", ["row_id"])
    assert table.frame["row_id"].tolist() == ["007", "8"]
    assert len(table.sha256 or "") == 64


def test_strict_feature_agreement() -> None:
    train = pd.DataFrame({"x": [1, 2, 3], "z": [1, 2, 3], "y": [1.0, 2, 3]})
    test = pd.DataFrame({"x": [1, 2, 3], "y": [1.0, 2, 3]})
    with pytest.raises(InputValidationError, match="disagree"):
        build_context(train=train, test=test, target="y", task="regression")
    from mldoctor.core.config import DoctorConfig

    ctx, _ = build_context(
        train=train, test=test, target="y", task="regression",
        config=DoctorConfig(allow_extra_columns=True),
    )
    assert ctx.feature_columns == ["x"]
    assert any("differ" in w for w in ctx.warnings)


def test_metadata_columns_are_not_features() -> None:
    train = pd.DataFrame(
        {"rid": ["1", "2"], "g": ["a", "b"], "x": [1.0, 2.0], "y": [0.0, 1.0]}
    )
    ctx, _ = build_context(train=train, target="y", task="regression", row_id="rid", group="g")
    assert ctx.feature_columns == ["x"]
    assert ctx.numeric_features == ["x"]


def test_predictions_require_row_id_and_targets() -> None:
    train = pd.DataFrame({"row_id": ["1", "2"], "x": [1.0, 2.0], "y": [0.0, 1.0]})
    test = pd.DataFrame({"row_id": ["3", "4"], "x": [1.0, 2.0], "y": [0.0, np.nan]})
    preds = pd.DataFrame({"row_id": ["3", "4"], "y_pred": [0.0, 1.0]})
    with pytest.raises(InputValidationError, match="row ID"):
        build_context(train=train, test=test, predictions=preds, target="y", task="regression")
    with pytest.raises(InputValidationError, match="missing on 1 evaluation row"):
        build_context(train=train, test=test, predictions=preds, target="y",
                      task="regression", row_id="row_id")


def test_regression_requires_numeric_target() -> None:
    train = pd.DataFrame({"x": [1.0, 2.0], "y": ["a", "b"]})
    with pytest.raises(InputValidationError, match="must be numeric"):
        build_context(train=train, target="y", task="regression")


def test_missing_group_values_rejected() -> None:
    train = pd.DataFrame({"g": ["a", None], "x": [1.0, 2.0], "y": [0.0, 1.0]})
    with pytest.raises(InputValidationError, match="group column"):
        build_context(train=train, target="y", task="regression", group="g")


def test_source_frames_are_not_mutated() -> None:
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1, 0, 1]})
    before = train.copy()
    ctx, _ = build_context(train=train, target="y", task="classification")
    pd.testing.assert_frame_equal(train, before)
    assert ctx.class_labels == ["0", "1"]
