"""Missingness and data-integrity diagnostic.

Computed separately for every available split:

* per-feature missing fraction ``sum(is_missing(x_ij)) / n`` (ordinary NA only);
* ±Infinity counts in numeric features, reported separately from NA;
* constant (one distinct non-missing value) and near-constant features;
* duplicate full rows (features + target) and duplicate feature rows;
* duplicate feature rows with *discordant* targets;
* text columns mixing numeric and non-numeric values;
* columns whose parsed type differs between train and test;
* missing target values.

Assumptions/limitations: descriptive only. Missingness is *not* assumed to be MCAR, and no
test of the missingness mechanism is performed. Duplicates are exact matches after CSV
parsing (``1`` and ``1.0`` in a numeric column are equal; text is case-sensitive).
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from mldoctor.core.context import DiagnosticContext
from mldoctor.core.protocols import BaseDiagnostic
from mldoctor.core.schemas import DiagnosticResult, Finding, Severity
from mldoctor.io.validation import is_numeric_feature
from mldoctor.recommendations.rules import make_finding
from mldoctor.statistics.metrics import sig

_TOP_N = 50


def column_kind(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series.dtype):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series.dtype):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return "datetime"
    return "text"


def numeric_share(series: pd.Series) -> tuple[int, int]:
    """(numeric-looking count, non-missing count) for a text column."""
    present = series.dropna()
    if present.empty:
        return 0, 0
    as_text = present.astype(str).str.strip()
    numeric = pd.to_numeric(as_text, errors="coerce")
    return int(numeric.notna().sum()), len(present)


def missingness_severity(fraction: float, warn: float, high: float) -> Severity | None:
    if fraction >= high:
        return "high"
    if fraction >= warn:
        return "medium"
    return None


class DataIntegrityDiagnostic(BaseDiagnostic):
    id = "data_integrity"
    title = "Missingness and data integrity"
    description = (
        "Missing values, invalid numbers, constant features, duplicates, mixed and "
        "inconsistent types, missing targets."
    )
    limitations: ClassVar[tuple[str, ...]] = (
        "Descriptive only: the missingness mechanism (MCAR/MAR/MNAR) is not tested or assumed.",
        "Duplicate detection uses exact equality after parsing; near-duplicates are not found.",
    )

    def run(self, context: DiagnosticContext) -> DiagnosticResult:
        findings: list[Finding] = []
        metrics: dict[str, Any] = {"splits": {}}
        features = context.feature_columns

        for split, frame in context.splits():
            split_metrics, split_findings = self._split(context, split, frame, features)
            metrics["splits"][split] = split_metrics
            findings.extend(split_findings)

        if context.test is not None:
            test = context.test
            mismatches = []
            for column in features:
                a, b = context.train[column], test[column]
                if a.isna().all() or b.isna().all():
                    continue
                kind_a, kind_b = column_kind(a), column_kind(b)
                if kind_a != kind_b:
                    mismatches.append({"feature": column, "train": kind_a, "test": kind_b})
            metrics["dtype_mismatches"] = len(mismatches)
            if mismatches:
                findings.append(
                    make_finding(
                        "DTYPE_MISMATCH",
                        "medium",
                        f"{len(mismatches)} feature(s) parse to different types in train and test",
                        {"features": mismatches[:_TOP_N], "count": len(mismatches)},
                    )
                )
        return self.result(findings, metrics)

    def _split(
        self, context: DiagnosticContext, split: str, frame: pd.DataFrame, features: list[str]
    ) -> tuple[dict[str, Any], list[Finding]]:
        cfg = context.config.integrity
        findings: list[Finding] = []
        n = len(frame)
        feats = frame[features]

        # --- missingness ---------------------------------------------------------------
        missing_counts = feats.isna().sum()
        fractions = (missing_counts / n).astype(float)
        by_severity: dict[str, dict[str, float]] = {"high": {}, "medium": {}}
        for column, fraction in fractions.items():
            sev = missingness_severity(
                float(fraction), cfg.missing_warning_fraction, cfg.missing_high_fraction
            )
            if sev is not None:
                by_severity[sev][str(column)] = sig(float(fraction))
        level: Severity
        for level in ("high", "medium"):
            flagged = by_severity[level]
            if not flagged:
                continue
            threshold = (
                cfg.missing_high_fraction if level == "high" else cfg.missing_warning_fraction
            )
            ranked = dict(sorted(flagged.items(), key=lambda kv: (-kv[1], kv[0]))[:_TOP_N])
            findings.append(
                make_finding(
                    "HIGH_MISSINGNESS",
                    level,
                    f"{split}: {len(flagged)} feature(s) have missing fraction ≥ {threshold:g}",
                    {
                        "split": split,
                        "threshold": threshold,
                        "feature_count": len(flagged),
                        "missing_fraction": ranked,
                        "n_rows": n,
                    },
                    limitations=["Heuristic threshold; missingness is not assumed to be random."],
                )
            )

        # --- invalid numeric values (±inf) ---------------------------------------------
        inf_counts: dict[str, dict[str, int]] = {}
        for column in features:
            series = frame[column]
            if is_numeric_feature(series):
                values = series.to_numpy(dtype=float, na_value=np.nan)
                pos = int(np.isposinf(values).sum())
                neg = int(np.isneginf(values).sum())
                if pos or neg:
                    inf_counts[column] = {"positive_infinity": pos, "negative_infinity": neg}
        if inf_counts:
            findings.append(
                make_finding(
                    "INVALID_NUMERIC_VALUES",
                    "medium",
                    f"{split}: {len(inf_counts)} feature(s) contain ±Infinity values",
                    {"split": split, "features": dict(list(inf_counts.items())[:_TOP_N])},
                )
            )

        # --- constant / near-constant ------------------------------------------------
        constant: list[str] = []
        near_constant: dict[str, float] = {}
        all_missing: list[str] = []
        for column in features:
            present = frame[column].dropna()
            if present.empty:
                all_missing.append(column)
                continue
            counts = present.value_counts()
            if len(counts) == 1:
                constant.append(column)
            else:
                top_share = float(counts.iloc[0] / len(present))
                if top_share >= cfg.near_constant_fraction:
                    near_constant[column] = sig(top_share)
        if constant:
            findings.append(
                make_finding(
                    "CONSTANT_FEATURE",
                    "low",
                    f"{split}: {len(constant)} feature(s) have a single observed value",
                    {"split": split, "features": constant[:_TOP_N], "count": len(constant)},
                )
            )
        if near_constant:
            findings.append(
                make_finding(
                    "NEAR_CONSTANT_FEATURE",
                    "info",
                    f"{split}: {len(near_constant)} feature(s) have one value in ≥ "
                    f"{cfg.near_constant_fraction:.0%} of non-missing rows",
                    {
                        "split": split,
                        "threshold": cfg.near_constant_fraction,
                        "top_value_share": dict(list(near_constant.items())[:_TOP_N]),
                    },
                )
            )

        # --- duplicates -----------------------------------------------------------------
        target_available = context.target in frame.columns
        dup_feature_rows = int(feats.duplicated().sum())
        dup_full_rows = (
            int(frame[[*features, context.target]].duplicated().sum())
            if target_available
            else None
        )
        discordant_groups = 0
        discordant_rows = 0
        if target_available and dup_feature_rows:
            dup_mask = feats.duplicated(keep=False)
            sub = frame.loc[dup_mask, [*features, context.target]]
            keys = pd.util.hash_pandas_object(sub[features], index=False)
            labelled = sub[context.target].notna()
            grouped = sub.loc[labelled, context.target].groupby(keys[labelled])
            n_targets = grouped.nunique()
            discordant = n_targets[n_targets > 1]
            discordant_groups = len(discordant)
            discordant_rows = int(keys[labelled].isin(discordant.index).sum())
        if dup_feature_rows:
            findings.append(
                make_finding(
                    "DUPLICATE_ROWS",
                    "low",
                    f"{split}: {dup_feature_rows} row(s) repeat an earlier feature row"
                    + (f" ({dup_full_rows} also repeat the target)" if dup_full_rows else ""),
                    {
                        "split": split,
                        "duplicate_feature_rows": dup_feature_rows,
                        "duplicate_full_rows": dup_full_rows,
                        "fraction": sig(dup_feature_rows / n),
                    },
                )
            )
        if discordant_groups:
            findings.append(
                make_finding(
                    "DISCORDANT_DUPLICATES",
                    "medium",
                    f"{split}: {discordant_groups} set(s) of identical feature rows have "
                    f"different targets ({discordant_rows} rows)",
                    {
                        "split": split,
                        "discordant_groups": discordant_groups,
                        "rows_involved": discordant_rows,
                    },
                    limitations=[
                        "For continuous targets any difference counts as discordant, including "
                        "measurement noise."
                    ],
                )
            )

        # --- mixed types -----------------------------------------------------------------
        mixed: dict[str, dict[str, Any]] = {}
        mixed_severity: Severity = "low"
        for column in features:
            series = frame[column]
            if column_kind(series) != "text":
                continue
            numeric_count, n_present = numeric_share(series)
            if n_present == 0 or numeric_count in (0, n_present):
                continue
            share = numeric_count / n_present
            if share < cfg.mixed_type_min_fraction:
                continue
            mixed[column] = {
                "numeric_share": sig(share),
                "non_numeric_count": n_present - numeric_count,
            }
            if share >= 0.5:
                mixed_severity = "medium"
        if mixed:
            findings.append(
                make_finding(
                    "MIXED_TYPES",
                    mixed_severity,
                    f"{split}: {len(mixed)} text column(s) mix numeric and non-numeric values",
                    {"split": split, "features": dict(list(mixed.items())[:_TOP_N])},
                    limitations=["Raw values are not reported to preserve row privacy."],
                )
            )

        # --- missing target ---------------------------------------------------------------
        target_missing = int(frame[context.target].isna().sum()) if target_available else None
        if target_missing:
            findings.append(
                make_finding(
                    "MISSING_TARGET",
                    "medium",
                    f"{split}: target '{context.target}' is missing on {target_missing} row(s)",
                    {
                        "split": split,
                        "missing_count": target_missing,
                        "fraction": sig(target_missing / n),
                    },
                )
            )

        total_cells = n * len(features)
        split_metrics: dict[str, Any] = {
            "n_rows": n,
            "n_features": len(features),
            "missing_cell_fraction": (
                sig(float(missing_counts.sum()) / total_cells) if total_cells else None
            ),
            "features_with_missing": int((missing_counts > 0).sum()),
            "all_missing_features": all_missing[:_TOP_N],
            "constant_features": len(constant),
            "near_constant_features": len(near_constant),
            "features_with_infinity": len(inf_counts),
            "duplicate_feature_rows": dup_feature_rows,
            "duplicate_full_rows": dup_full_rows,
            "discordant_duplicate_groups": discordant_groups,
            "mixed_type_features": len(mixed),
            "target_missing": target_missing,
        }
        return split_metrics, findings
