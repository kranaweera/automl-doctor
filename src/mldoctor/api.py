"""Public Python API.

Example::

    from mldoctor.api import diagnose

    report = diagnose(
        train="train.csv", test="test.csv", predictions="test_predictions.csv",
        target="outcome", task="regression", row_id="row_id",
    )
    print(report.summary)
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from mldoctor.core.config import DoctorConfig, build_config
from mldoctor.core.context import DiagnosticContext
from mldoctor.core.engine import build_report, run_diagnostics
from mldoctor.core.registry import DiagnosticRegistry, default_registry
from mldoctor.core.schemas import InputFileInfo, Report, Task
from mldoctor.io.csv_loader import LoadedTable, from_frame, load_csv
from mldoctor.io.prediction_loader import align_predictions
from mldoctor.io.validation import (
    InputValidationError,
    canonical_labels,
    is_numeric_feature,
    require_columns,
    resolve_feature_columns,
    validate_group,
    validate_target_for_evaluation,
    validate_unique_ids,
)

TableInput = str | Path | pd.DataFrame

DATA_ONLY_DIAGNOSTICS = ["data_integrity", "class_imbalance", "feature_correlation", "outliers"]
LEAKAGE_DIAGNOSTICS = ["leakage"]


def _load(source: TableInput, role: str, string_columns: list[str]) -> LoadedTable:
    if isinstance(source, pd.DataFrame):
        return from_frame(source, role, string_columns)
    return load_csv(source, role, string_columns)


def _label_sort_key(label: str) -> tuple[int, float, str]:
    try:
        return (0, float(label), label)
    except ValueError:
        return (1, 0.0, label)


def build_context(
    *,
    train: TableInput,
    target: str,
    task: Task,
    test: TableInput | None = None,
    predictions: TableInput | None = None,
    train_predictions: TableInput | None = None,
    row_id: str | None = None,
    group: str | None = None,
    config: DoctorConfig | None = None,
) -> tuple[DiagnosticContext, list[InputFileInfo]]:
    """Load and validate inputs, returning a :class:`DiagnosticContext`.

    Raises:
        InputValidationError: if any input contract is violated.
    """
    cfg = config or DoctorConfig()
    if task not in ("classification", "regression"):
        raise InputValidationError(f"task must be 'classification' or 'regression', got {task!r}")
    if (predictions is not None or train_predictions is not None) and row_id is None:
        raise InputValidationError(
            "a row ID column (--row-id) is required to join predictions; "
            "predictions are never matched by row position"
        )
    if predictions is not None and test is None:
        raise InputValidationError("held-out predictions were supplied without a test split")
    if len({c for c in (target, row_id, group, cfg.time_column) if c is not None}) != len(
        [c for c in (target, row_id, group, cfg.time_column) if c is not None]
    ):
        raise InputValidationError("target, row ID, group and time columns must be distinct")

    warnings: list[str] = []
    id_columns = [c for c in (row_id, group) if c is not None]
    train_table = _load(train, "train", id_columns)
    test_table = _load(test, "test", id_columns) if test is not None else None
    pred_table = _load(predictions, "predictions", id_columns) if predictions is not None else None
    train_pred_table = (
        _load(train_predictions, "train_predictions", id_columns)
        if train_predictions is not None
        else None
    )

    train_df = train_table.frame.copy()
    test_df = test_table.frame.copy() if test_table is not None else None

    require_columns(train_df, [target, row_id, group, cfg.time_column], "train")
    test_has_target = False
    if test_df is not None:
        require_columns(test_df, [row_id, group, cfg.time_column], "test")
        test_has_target = target in test_df.columns
        if not test_has_target:
            if pred_table is not None:
                raise InputValidationError(
                    f"test: target column '{target}' is required to evaluate predictions"
                )
            warnings.append(f"test split has no '{target}' column; label-based checks skipped")

    for name in cfg.exclude_columns:
        if name not in train_df.columns:
            warnings.append(f"excluded column '{name}' not found in train")
    metadata = {c for c in (target, row_id, group, cfg.time_column) if c is not None}
    metadata |= set(cfg.exclude_columns)
    features, feature_warnings = resolve_feature_columns(
        train_df, test_df, metadata, cfg.allow_extra_columns
    )
    warnings.extend(feature_warnings)

    if group is not None:
        validate_group(train_df, group, "train")
        if test_df is not None:
            validate_group(test_df, group, "test")

    if row_id is not None:
        for role, frame in (("train", train_df), ("test", test_df)):
            if frame is None:
                continue
            needs_join = (role == "test" and pred_table is not None) or (
                role == "train" and train_pred_table is not None
            )
            if needs_join:
                validate_unique_ids(frame, row_id, role)
            elif frame[row_id].isna().any() or frame[row_id].duplicated().any():
                warnings.append(f"{role}: row ID column '{row_id}' is not unique/complete")

    # --- target handling ---------------------------------------------------------------
    class_labels: list[str] | None = None
    if task == "regression":
        for role, frame in (("train", train_df), ("test", test_df if test_has_target else None)):
            if frame is not None and not is_numeric_feature(frame[target]):
                raise InputValidationError(
                    f"{role}: regression target '{target}' must be numeric "
                    f"(parsed dtype {frame[target].dtype})"
                )
        n_unique = int(train_df[target].nunique(dropna=True))
        if n_unique <= 2:
            warnings.append(
                f"regression target has only {n_unique} distinct value(s); is this classification?"
            )
    else:
        train_df[target] = canonical_labels(train_df[target])
        labels = set(train_df[target].dropna().tolist())
        if test_df is not None and test_has_target:
            test_df[target] = canonical_labels(test_df[target])
            labels |= set(test_df[target].dropna().tolist())
        class_labels = sorted(labels, key=_label_sort_key)
        if not class_labels:
            raise InputValidationError(f"train: target '{target}' has no non-missing values")
        if len(class_labels) > max(50, len(train_df) // 2):
            warnings.append(
                f"classification target has {len(class_labels)} distinct labels; "
                "is this a regression task?"
            )

    # --- predictions ---------------------------------------------------------------------
    test_pred = None
    if pred_table is not None and test_df is not None and row_id is not None:
        validate_target_for_evaluation(test_df, target, task, "test")
        test_pred = align_predictions(
            pred_table.frame, test_df, row_id, task, "predictions", class_labels
        )
    train_pred = None
    if train_pred_table is not None and row_id is not None:
        validate_target_for_evaluation(train_df, target, task, "train")
        train_pred = align_predictions(
            train_pred_table.frame, train_df, row_id, task, "train_predictions", class_labels
        )

    numeric = [
        c
        for c in features
        if is_numeric_feature(train_df[c]) and (test_df is None or is_numeric_feature(test_df[c]))
    ]

    context = DiagnosticContext(
        task=task,
        target=target,
        train=train_df,
        test=test_df,
        feature_columns=features,
        numeric_features=numeric,
        config=cfg,
        row_id=row_id,
        group=group,
        test_predictions=test_pred,
        train_predictions=train_pred,
        class_labels=class_labels,
        test_has_target=test_has_target,
        warnings=warnings,
    )
    inputs = [
        InputFileInfo(
            role=t.role, sha256=t.sha256, rows=len(t.frame), columns=len(t.frame.columns)
        )
        for t in (train_table, test_table, pred_table, train_pred_table)
        if t is not None
    ]
    return context, inputs


def diagnose(
    *,
    train: TableInput,
    target: str,
    task: Task,
    test: TableInput | None = None,
    predictions: TableInput | None = None,
    train_predictions: TableInput | None = None,
    row_id: str | None = None,
    group: str | None = None,
    config: DoctorConfig | dict[str, Any] | None = None,
    diagnostics: list[str] | None = None,
    registry: DiagnosticRegistry | None = None,
    command: str = "diagnose",
) -> Report:
    """Validate inputs, run diagnostics and return a validated :class:`Report`."""
    started = datetime.now(UTC)
    t0 = time.perf_counter()
    cfg = config if isinstance(config, DoctorConfig) else build_config(overrides=config)
    context, inputs = build_context(
        train=train,
        test=test,
        predictions=predictions,
        train_predictions=train_predictions,
        target=target,
        task=task,
        row_id=row_id,
        group=group,
        config=cfg,
    )
    reg = registry or default_registry(load_plugins=cfg.load_plugins)
    selected = reg.select(diagnostics if diagnostics is not None else cfg.enabled_diagnostics)
    results = run_diagnostics(selected, context)
    return build_report(
        context,
        results,
        started_at=started,
        command=command,
        inputs=inputs,
        duration_ms=(time.perf_counter() - t0) * 1000.0,
    )
