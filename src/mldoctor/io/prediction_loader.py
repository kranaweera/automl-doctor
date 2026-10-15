"""Prediction loading and ID-based alignment.

Predictions are joined to evaluation rows **by row ID, never by position**. Any of the
following is a contract violation (``InputValidationError``):

* duplicate or missing prediction row IDs;
* evaluation rows without a prediction, or predictions for unknown rows;
* missing / non-finite (regression) predictions;
* predicted class labels that do not occur in the declared labels (classification).

Only hard labels are supported in v0.1; probability columns are out of scope.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from mldoctor.io.validation import (
    InputValidationError,
    canonical_labels,
    is_numeric_feature,
    require_columns,
    validate_unique_ids,
)

PREDICTION_COLUMN = "y_pred"


def validate_prediction_frame(frame: pd.DataFrame, row_id: str, role: str) -> None:
    require_columns(frame, [row_id, PREDICTION_COLUMN], role)
    validate_unique_ids(frame, row_id, role)
    n_missing = int(frame[PREDICTION_COLUMN].isna().sum())
    if n_missing:
        raise InputValidationError(f"{role}: {n_missing} missing value(s) in '{PREDICTION_COLUMN}'")


def align_predictions(
    predictions: pd.DataFrame,
    evaluation: pd.DataFrame,
    row_id: str,
    task: str,
    role: str,
    class_labels: Sequence[str] | None = None,
) -> pd.Series:
    """Return predictions re-indexed to ``evaluation``'s index via ``row_id``.

    For classification the returned labels are canonical strings (see
    :func:`mldoctor.io.validation.canonical_labels`); for regression they are floats.
    """
    validate_prediction_frame(predictions, row_id, role)
    eval_ids = evaluation[row_id]
    pred_ids = predictions[row_id]
    eval_set = set(eval_ids.tolist())
    pred_set = set(pred_ids.tolist())
    missing = eval_set - pred_set
    unexpected = pred_set - eval_set
    if missing or unexpected:
        raise InputValidationError(
            f"{role}: prediction IDs do not match evaluation rows "
            f"({len(missing)} evaluation row(s) without a prediction, "
            f"{len(unexpected)} prediction(s) for unknown row IDs). "
            "Predictions are joined by row ID, never by position."
        )
    lookup = pd.Series(predictions[PREDICTION_COLUMN].to_numpy(), index=pred_ids.to_numpy())
    aligned = pd.Series(lookup.loc[eval_ids.to_numpy()].to_numpy(), index=evaluation.index)

    if task == "regression":
        if not is_numeric_feature(aligned):
            coerced = pd.to_numeric(aligned, errors="coerce")
            n_bad = int(coerced.isna().sum())
            if n_bad:
                raise InputValidationError(
                    f"{role}: {n_bad} non-numeric regression prediction(s) in '{PREDICTION_COLUMN}'"
                )
            aligned = coerced
        values = aligned.astype(float)
        n_nonfinite = int((~np.isfinite(values.to_numpy())).sum())
        if n_nonfinite:
            raise InputValidationError(
                f"{role}: {n_nonfinite} non-finite regression prediction(s) (NaN/±Inf)"
            )
        return values

    labels = canonical_labels(aligned)
    if class_labels is not None:
        unknown = sorted(set(labels.tolist()) - set(class_labels))
        if unknown:
            raise InputValidationError(
                f"{role}: {len(unknown)} predicted label value(s) not among the declared class "
                f"labels {list(class_labels)}: {unknown[:10]}"
            )
    return labels
