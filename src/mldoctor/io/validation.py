"""Input-contract validation.

Contract violations raise :class:`InputValidationError` (CLI exit code 2). They are *not*
diagnostic findings: a report built from invalid inputs would be misleading, so the run
stops instead of returning a "successful" report.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


class InputValidationError(ValueError):
    """Raised when inputs violate the documented data contract."""


def is_numeric_feature(series: pd.Series) -> bool:
    """Numeric, non-boolean dtype (booleans are treated as categorical)."""
    return bool(
        pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(series.dtype)
    )


def canonical_labels(series: pd.Series) -> pd.Series:
    """Map class labels to canonical strings so that ``1``, ``1.0`` and ``"1"`` agree.

    Numeric series whose finite values are all integral are rendered as integers;
    other numeric values use ``repr(float)``. Missing values stay missing (``None``).
    """
    if pd.api.types.is_bool_dtype(series.dtype):
        return _object_series([None if pd.isna(v) else str(bool(v)) for v in series], series)
    if pd.api.types.is_numeric_dtype(series.dtype):
        values = series.astype(float)
        finite = values[np.isfinite(values)]
        integral = bool(len(finite) == 0 or np.all(np.equal(np.mod(finite, 1), 0)))

        def render(v: float) -> str | None:
            if pd.isna(v):
                return None
            if integral and np.isfinite(v):
                return str(int(v))
            return repr(float(v))

        return _object_series([render(v) for v in values], series)

    def render_obj(v: object) -> str | None:
        if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
            return None
        text = str(v)
        # "1.0" written as text in one file and 1 in another: normalise integral floats.
        try:
            number = float(text)
        except ValueError:
            return text
        if np.isfinite(number) and number.is_integer() and text.strip() != "":
            return str(int(number))
        return text

    return _object_series([render_obj(v) for v in series], series)


def _object_series(values: list[str | None], like: pd.Series) -> pd.Series:
    # Built explicitly so that missing labels stay ``None`` rather than becoming NaN.
    return pd.Series(values, index=like.index, dtype=object, name=like.name)


def require_columns(frame: pd.DataFrame, columns: Iterable[str | None], role: str) -> None:
    missing = [c for c in columns if c is not None and c not in frame.columns]
    if missing:
        raise InputValidationError(
            f"{role}: missing required column(s) {missing}; available: {list(frame.columns)}"
        )


def validate_unique_ids(frame: pd.DataFrame, row_id: str, role: str) -> None:
    ids = frame[row_id]
    n_missing = int(ids.isna().sum())
    if n_missing:
        raise InputValidationError(f"{role}: {n_missing} row(s) have a missing '{row_id}'")
    n_dupes = int(ids.duplicated().sum())
    if n_dupes:
        raise InputValidationError(
            f"{role}: '{row_id}' must be unique within the split; {n_dupes} duplicate value(s)"
        )


def validate_group(frame: pd.DataFrame, group: str, role: str) -> None:
    n_missing = int(frame[group].isna().sum())
    if n_missing:
        raise InputValidationError(
            f"{role}: group column '{group}' has {n_missing} missing value(s); every row must "
            "belong to a group for group-aware checks"
        )


def validate_target_for_evaluation(
    frame: pd.DataFrame, target: str, task: str, role: str
) -> None:
    """Targets on rows that will be scored must be present (and finite for regression)."""
    y = frame[target]
    n_missing = int(y.isna().sum())
    if n_missing:
        raise InputValidationError(
            f"{role}: target '{target}' is missing on {n_missing} evaluation row(s)"
        )
    if task == "regression":
        if not is_numeric_feature(y):
            raise InputValidationError(
                f"{role}: regression target '{target}' must be numeric (dtype {y.dtype})"
            )
        n_nonfinite = int((~np.isfinite(y.astype(float))).sum())
        if n_nonfinite:
            raise InputValidationError(
                f"{role}: regression target '{target}' has {n_nonfinite} non-finite value(s)"
            )


def resolve_feature_columns(
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    metadata: set[str],
    allow_extra_columns: bool,
) -> tuple[list[str], list[str]]:
    """Return (feature columns, warnings).

    Metadata columns (target, row ID, group, time, explicit excludes) are never features.
    With the default strict policy train and test must expose exactly the same features.
    """
    warnings: list[str] = []
    train_features = [c for c in train.columns if c not in metadata]
    if test is None:
        features = train_features
    else:
        test_features = [c for c in test.columns if c not in metadata]
        only_train = sorted(set(train_features) - set(test_features))
        only_test = sorted(set(test_features) - set(train_features))
        if only_train or only_test:
            if not allow_extra_columns:
                raise InputValidationError(
                    "train/test feature columns disagree (strict mode): "
                    f"only in train={only_train}, only in test={only_test}. "
                    "Set allow_extra_columns=true to use the shared columns only."
                )
            warnings.append(
                f"feature columns differ; using {len(set(train_features) & set(test_features))} "
                f"shared columns (dropped train-only={only_train}, test-only={only_test})"
            )
        features = [c for c in train_features if c in set(test_features)]
    if not features:
        raise InputValidationError("no feature columns remain after removing metadata columns")
    return features, warnings
