"""Leakage *indicators*.

Sub-checks:

1. ``cross_split_overlap`` – exact feature-row matches between train and test using a
   stable canonical serialization (numbers as ``repr(float)`` with ``-0.0 → 0.0``, text
   verbatim, missing values as an explicit NA token so that NA matches NA). Counts and
   fractions are reported on both sides.
2. ``group_overlap`` – intersection of group IDs across splits. Severity depends on the
   declared evaluation target: ``critical`` for ``unseen_groups``, ``medium`` when the goal
   is ``unspecified``, ``info`` for ``unseen_rows``.
3. ``target_copy`` – a feature that equals the target, is an exact linear transform of it
   (regression, |r| = 1), or is a one-to-one recoding of it (classification).
4. ``target_association`` – warning-only screen for near-perfect feature–target association:
   |Pearson| or |Spearman| ≥ 0.98 (regression), binary AUC ≥ 0.99 (numeric features,
   binary classification), class purity ≥ 0.99 for low-cardinality features
   (classification) or correlation ratio η ≥ 0.98 for categorical features (regression).
5. ``suspicious_names`` – feature names matching configurable patterns or containing the
   target name.

Limitations: these checks cannot establish or rule out pipeline/preprocessing leakage
(e.g. scalers, imputers or feature selection fit on all data, temporal look-ahead).
Passing them is *not* evidence that no leakage exists.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from mldoctor.core.context import DiagnosticContext
from mldoctor.core.protocols import BaseDiagnostic
from mldoctor.core.schemas import DiagnosticResult, Finding, Severity
from mldoctor.io.validation import canonical_labels, is_numeric_feature
from mldoctor.recommendations.rules import make_finding
from mldoctor.statistics.correlations import pearson_spearman
from mldoctor.statistics.metrics import sig

NA_TOKEN = "\x00NA"
_SEP = "\x1f"
LOW_CARDINALITY_MAX = 20
PIPELINE_LIMITATION = (
    "Pipeline/preprocessing leakage (e.g. scaling, imputation or feature selection fit on "
    "all data, temporal look-ahead) cannot be detected from CSV files; passing these checks "
    "does not rule out leakage."
)


def canonical_column(series: pd.Series) -> list[str]:
    """Stable, type-tagged string form of each value (NA preserved as a token)."""
    if pd.api.types.is_bool_dtype(series.dtype):
        return [NA_TOKEN if pd.isna(v) else f"b:{bool(v)}" for v in series]
    if is_numeric_feature(series):
        out: list[str] = []
        for v in series.to_numpy(dtype=float, na_value=np.nan):
            if np.isnan(v):
                out.append(NA_TOKEN)
            else:
                out.append("n:" + repr(0.0 if v == 0 else float(v)))
        return out
    return [NA_TOKEN if pd.isna(v) else f"s:{v}" for v in series]


def canonical_row_keys(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    cols = [canonical_column(frame[c]) for c in columns]
    return [_SEP.join(parts) for parts in zip(*cols, strict=True)] if cols else []


def binary_auc(scores: np.ndarray, positive: np.ndarray) -> float | None:
    """Mann–Whitney AUC = P(score_pos > score_neg) + ½ P(tie); None if a class is empty."""
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(scores)
    u = float(ranks[positive].sum()) - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def correlation_ratio(categories: np.ndarray, values: np.ndarray) -> float | None:
    """η = sqrt(SS_between / SS_total) of ``values`` across ``categories``; None if SS_total=0."""
    total = float(np.sum((values - values.mean()) ** 2))
    if total == 0:
        return None
    frame = pd.DataFrame({"c": categories, "v": values})
    stats = frame.groupby("c")["v"].agg(["mean", "size"])
    between = float(np.sum(stats["size"] * (stats["mean"] - values.mean()) ** 2))
    return float(np.sqrt(min(between / total, 1.0)))


class LeakageDiagnostic(BaseDiagnostic):
    id = "leakage"
    title = "Data leakage indicators"
    description = (
        "Cross-split duplicate rows, group overlap, target copies, near-perfect target "
        "association and suspicious feature names."
    )
    limitations: ClassVar[tuple[str, ...]] = (PIPELINE_LIMITATION,)

    def run(self, context: DiagnosticContext) -> DiagnosticResult:
        findings: list[Finding] = []
        metrics: dict[str, Any] = {}
        skipped: dict[str, str] = {}

        if context.test is None:
            skipped["cross_split_overlap"] = "no test split supplied"
            skipped["group_overlap"] = "no test split supplied"
        else:
            self._cross_split(context, findings, metrics)
            if context.group is None:
                skipped["group_overlap"] = "no group column declared (--group)"
            else:
                self._group_overlap(context, findings, metrics)

        self._target_checks(context, findings, metrics, skipped)
        self._names(context, findings, metrics)
        metrics["evaluation_target"] = context.config.evaluation_target
        return self.result(findings, metrics, skipped)

    # -- 1. cross-split overlap ---------------------------------------------------------
    def _cross_split(
        self, ctx: DiagnosticContext, findings: list[Finding], metrics: dict[str, Any]
    ) -> None:
        assert ctx.test is not None
        cfg = ctx.config.leakage
        features = ctx.feature_columns
        train_keys = canonical_row_keys(ctx.train, features)
        test_keys = canonical_row_keys(ctx.test, features)
        train_set, test_set = set(train_keys), set(test_keys)
        test_in_train = sum(1 for k in test_keys if k in train_set)
        train_in_test = sum(1 for k in train_keys if k in test_set)
        n_train, n_test = len(train_keys), len(test_keys)
        distinct_train = len(train_set)

        same_target: int | None = None
        if ctx.test_has_target and test_in_train:
            train_pairs = set(
                zip(train_keys, canonical_column(ctx.train[ctx.target]), strict=True)
            )
            test_targets = canonical_column(ctx.test[ctx.target])
            same_target = sum(
                1
                for key, y in zip(test_keys, test_targets, strict=True)
                if key in train_set and (key, y) in train_pairs
            )

        test_fraction = test_in_train / n_test if n_test else 0.0
        low_cardinality = distinct_train < 0.5 * n_train
        metrics["cross_split_overlap"] = {
            "test_rows_in_train": test_in_train,
            "test_fraction": sig(test_fraction),
            "train_rows_in_test": train_in_test,
            "train_fraction": sig(train_in_test / n_train) if n_train else 0.0,
            "overlapping_distinct_rows": len(train_set & test_set),
            "test_rows_with_same_target": same_target,
            "distinct_train_feature_rows": distinct_train,
            "low_cardinality_feature_space": low_cardinality,
        }
        if not test_in_train:
            return
        severity: Severity
        limitations = [
            "Exact matching only; near-duplicates (rounding, reordered text) are not detected."
        ]
        if low_cardinality:
            severity = "low"
            limitations.append(
                f"Only {distinct_train} distinct feature rows among {n_train} training rows: "
                "exact matches across splits are expected by chance in such a small feature "
                "space and are weak evidence of contamination."
            )
        elif test_fraction >= cfg.cross_split_high_fraction:
            severity = "high"
        else:
            severity = "medium"
        findings.append(
            make_finding(
                "CROSS_SPLIT_DUPLICATE",
                severity,
                f"{test_in_train} of {n_test} test rows ({test_fraction:.1%}) have a feature row "
                f"identical to a training row; {train_in_test} training rows appear in test",
                metrics["cross_split_overlap"],
                limitations=limitations,
            )
        )

    # -- 2. group overlap ----------------------------------------------------------------
    def _group_overlap(
        self, ctx: DiagnosticContext, findings: list[Finding], metrics: dict[str, Any]
    ) -> None:
        assert ctx.test is not None and ctx.group is not None
        train_groups = set(ctx.train[ctx.group].tolist())
        test_groups_series = ctx.test[ctx.group]
        test_groups = set(test_groups_series.tolist())
        shared = train_groups & test_groups
        rows_affected = int(test_groups_series.isin(shared).sum())
        evidence = {
            "overlapping_groups": len(shared),
            "train_groups": len(train_groups),
            "test_groups": len(test_groups),
            "test_group_fraction": sig(len(shared) / len(test_groups)),
            "test_rows_in_overlapping_groups": rows_affected,
            "test_row_fraction": sig(rows_affected / len(ctx.test)),
            "evaluation_target": ctx.config.evaluation_target,
        }
        metrics["group_overlap"] = evidence
        if not shared:
            return
        goal = ctx.config.evaluation_target
        severity: Severity
        if goal == "unseen_groups":
            severity = "critical"
            note = "the declared goal is generalization to unseen groups"
        elif goal == "unspecified":
            severity = "medium"
            note = (
                "declare --evaluation-target unseen_groups or unseen_rows to calibrate severity"
            )
        else:
            severity = "info"
            note = "acceptable for the declared unseen-row goal, but rows are not independent"
        findings.append(
            make_finding(
                "GROUP_OVERLAP",
                severity,
                f"{len(shared)} group(s) appear in both splits "
                f"({rows_affected} test rows); {note}",
                evidence,
                limitations=["Group identifiers themselves are not reported."],
            )
        )

    # -- 3/4. target copy and association ------------------------------------------------
    def _target_checks(
        self,
        ctx: DiagnosticContext,
        findings: list[Finding],
        metrics: dict[str, Any],
        skipped: dict[str, str],
    ) -> None:
        cfg = ctx.config.leakage
        frame = ctx.train
        y = frame[ctx.target]
        labelled = y.notna().to_numpy()
        copies: list[dict[str, Any]] = []
        associations: list[dict[str, Any]] = []
        multiclass_numeric: list[str] = []

        for column in ctx.feature_columns:
            x = frame[column]
            mask = labelled & x.notna().to_numpy()
            n = int(mask.sum())
            if n < cfg.min_n:
                continue
            xs, ys = x[mask], y[mask]
            numeric = is_numeric_feature(x)

            if ctx.task == "regression":
                y_values = ys.to_numpy(dtype=float)
                if numeric:
                    x_values = xs.to_numpy(dtype=float)
                    finite = np.isfinite(x_values) & np.isfinite(y_values)
                    if finite.sum() < cfg.min_n:
                        continue
                    if np.array_equal(x_values[finite], y_values[finite]):
                        copies.append({"feature": column, "kind": "exact_copy", "n": n})
                        continue
                    n_pair, r, rho = pearson_spearman(x_values, y_values)
                    if r is not None and abs(r) >= 1 - 1e-12:
                        copies.append(
                            {"feature": column, "kind": "exact_linear_transform", "n": n_pair,
                             "pearson": sig(r)}
                        )
                        continue
                    strongest = max(abs(r or 0.0), abs(rho or 0.0))
                    if strongest >= cfg.association_threshold:
                        associations.append(
                            {"feature": column, "measure": "max(|pearson|,|spearman|)",
                             "value": sig(strongest), "n": n_pair,
                             "pearson": None if r is None else sig(r),
                             "spearman": None if rho is None else sig(rho)}
                        )
                else:
                    eta = correlation_ratio(canonical_labels(xs).to_numpy(), y_values)
                    n_distinct = int(xs.nunique())
                    if (
                        eta is not None
                        and 2 <= n_distinct <= n / 5
                        and eta >= cfg.association_threshold
                    ):
                        associations.append(
                            {"feature": column, "measure": "correlation_ratio_eta",
                             "value": sig(eta), "n": n, "distinct_values": n_distinct}
                        )
                continue

            # classification: compare on canonical labels
            x_labels = canonical_labels(xs)
            y_labels = ys.astype(str)
            n_distinct = int(x_labels.nunique())
            n_classes = int(y_labels.nunique())
            if n_classes < 2:
                continue
            if (x_labels.to_numpy() == y_labels.to_numpy()).all():
                copies.append({"feature": column, "kind": "exact_copy", "n": n})
                continue
            if n_distinct == n_classes:
                pairs = pd.DataFrame({"x": x_labels.to_numpy(), "y": y_labels.to_numpy()})
                if (
                    pairs.groupby("x")["y"].nunique().max() == 1
                    and pairs.groupby("y")["x"].nunique().max() == 1
                ):
                    copies.append({"feature": column, "kind": "one_to_one_recoding", "n": n})
                    continue
            if not numeric or n_distinct <= LOW_CARDINALITY_MAX:
                if n_distinct < 2 or n_distinct > n / 5:
                    continue
                table = pd.crosstab(x_labels.to_numpy(), y_labels.to_numpy())
                purity = float(table.max(axis=1).sum() / n)
                baseline = float(y_labels.value_counts().iloc[0] / n)
                if purity >= cfg.purity_threshold and baseline < cfg.purity_threshold:
                    associations.append(
                        {"feature": column, "measure": "class_purity", "value": sig(purity),
                         "baseline_majority_share": sig(baseline), "n": n,
                         "distinct_values": n_distinct}
                    )
                continue
            if n_classes == 2:
                positive = (y_labels == sorted(y_labels.unique())[1]).to_numpy()
                auc = binary_auc(xs.to_numpy(dtype=float), positive)
                if auc is not None and max(auc, 1 - auc) >= cfg.auc_threshold:
                    associations.append(
                        {"feature": column, "measure": "auc", "value": sig(auc), "n": n}
                    )
            elif n_classes > 2:
                multiclass_numeric.append(column)

        if multiclass_numeric:
            skipped["target_association_multiclass_numeric"] = (
                f"{len(multiclass_numeric)} continuous feature(s) not screened: the v0.1 "
                "association screen for continuous features supports binary targets only"
            )
        metrics["target_copy_features"] = [c["feature"] for c in copies]
        metrics["suspicious_associations"] = len(associations)
        if copies:
            findings.append(
                make_finding(
                    "POSSIBLE_TARGET_COPY",
                    "high",
                    f"{len(copies)} feature(s) reproduce or deterministically encode the target "
                    "in training data",
                    {"features": copies},
                )
            )
        if associations:
            findings.append(
                make_finding(
                    "SUSPICIOUS_TARGET_ASSOCIATION",
                    "medium",
                    f"{len(associations)} feature(s) have near-perfect association with the "
                    "target in training data (screen, not proof of leakage)",
                    {
                        "features": associations,
                        "thresholds": {
                            "correlation": cfg.association_threshold,
                            "auc": cfg.auc_threshold,
                            "purity": cfg.purity_threshold,
                        },
                    },
                    limitations=[
                        "A strong legitimate predictor produces the same signal; this is a prompt "
                        "to verify availability at prediction time."
                    ],
                )
            )

    # -- 5. names ------------------------------------------------------------------------
    def _names(
        self, ctx: DiagnosticContext, findings: list[Finding], metrics: dict[str, Any]
    ) -> None:
        raw_patterns = ctx.config.leakage.suspicious_name_patterns
        patterns = [re.compile(p, re.IGNORECASE) for p in raw_patterns]
        target_lower = ctx.target.lower()
        matches: list[dict[str, str]] = []
        for column in ctx.feature_columns:
            lower = column.lower()
            if len(target_lower) >= 3 and target_lower in lower:
                matches.append({"feature": column, "reason": "contains target name"})
                continue
            for pattern in patterns:
                if pattern.search(column):
                    matches.append({"feature": column, "reason": f"matches /{pattern.pattern}/"})
                    break
        metrics["suspicious_names"] = len(matches)
        if matches:
            findings.append(
                make_finding(
                    "SUSPICIOUS_FEATURE_NAME",
                    "low",
                    f"{len(matches)} feature name(s) resemble the target or post-outcome fields",
                    {"features": matches},
                )
            )
