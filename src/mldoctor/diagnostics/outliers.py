"""Univariate outlier screening for numeric features.

Reference statistics (median, MAD, quartiles) come from the **training** split and are
applied unchanged to the test split, so test candidates are measured on the training
scale. Method per feature (see :mod:`mldoctor.statistics.robust`):

1. ``modified_z``: |0.6745 (x − median) / MAD| > 3.5 when MAD > 0;
2. ``iqr``: outside [Q1 − 1.5 IQR, Q3 + 1.5 IQR] when MAD = 0 but IQR > 0;
3. ``degenerate``: MAD = IQR = 0 — values differing from the median are counted, but no
   outlier claim is made.

Severity per split: ``medium`` if any value is *extreme* (|z| > ``extreme_z`` (10) or more
than 3 IQRs beyond the quartiles), ``low`` if some feature's candidate fraction is ≥ 1%,
otherwise ``info``. Candidates are for review; nothing is removed.
"""

from __future__ import annotations

from typing import Any, ClassVar

from mldoctor.core.context import DiagnosticContext
from mldoctor.core.protocols import BaseDiagnostic
from mldoctor.core.schemas import DiagnosticResult, Finding, Severity
from mldoctor.recommendations.rules import make_finding
from mldoctor.statistics.metrics import name_list, sig
from mldoctor.statistics.robust import OutlierScreen, SplitScreen, screen_outliers

EXTREME_Z = 10.0
FAR_OUT_IQR = 3.0
LOW_FRACTION = 0.01


class OutlierDiagnostic(BaseDiagnostic):
    id = "outliers"
    title = "Outlier screening"
    description = "Robust modified z-scores with IQR fallback, train-referenced."
    limitations: ClassVar[tuple[str, ...]] = (
        "Univariate screening only; multivariate or contextual anomalies are not detected.",
        "Heavy-tailed or multimodal features naturally produce candidates; they are not errors "
        "by default.",
    )

    def applicability(self, context: DiagnosticContext) -> str | None:
        base = super().applicability(context)
        if base is not None:
            return base
        if not context.numeric_features:
            return "no numeric features"
        return None

    def run(self, context: DiagnosticContext) -> DiagnosticResult:
        cfg = context.config.outliers
        screens: dict[str, OutlierScreen] = {}
        for column in context.numeric_features:
            test_values = context.test[column] if context.test is not None else None
            screens[column] = screen_outliers(
                context.train[column],
                test_values,
                z_threshold=cfg.modified_z_threshold,
                iqr_k=cfg.iqr_multiplier,
                use_iqr_fallback=cfg.use_iqr_fallback,
                min_n=cfg.min_n,
                extreme_z=EXTREME_Z,
                far_out_k=FAR_OUT_IQR,
            )

        methods: dict[str, int] = {}
        for screen in screens.values():
            methods[screen.method] = methods.get(screen.method, 0) + 1

        findings: list[Finding] = []
        split_metrics: dict[str, Any] = {}
        for split in ("train", "test"):
            if split == "test" and context.test is None:
                continue
            per_feature: list[dict[str, Any]] = []
            extreme_total = 0
            for column, screen in screens.items():
                part: SplitScreen | None = screen.train if split == "train" else screen.test
                if part is None or part.candidates == 0 or screen.method not in (
                    "modified_z", "iqr"
                ):
                    continue
                extreme_total += part.extreme
                per_feature.append(
                    {
                        "feature": column,
                        "method": screen.method,
                        "n": part.n,
                        "candidates": part.candidates,
                        "fraction": sig(part.candidates / part.n),
                        "extreme": part.extreme,
                        "max_score": None if part.max_score is None else sig(part.max_score),
                    }
                )
            per_feature.sort(key=lambda d: (-d["fraction"], d["feature"]))
            split_metrics[split] = {
                "features_with_candidates": len(per_feature),
                "total_candidates": sum(d["candidates"] for d in per_feature),
                "extreme_candidates": extreme_total,
            }
            if not per_feature:
                continue
            severity: Severity
            if extreme_total > 0:
                severity = "medium"
            elif any(d["fraction"] >= LOW_FRACTION for d in per_feature):
                severity = "low"
            else:
                severity = "info"
            reference = " (scored against training median/MAD)" if split == "test" else ""
            findings.append(
                make_finding(
                    "OUTLIER_CANDIDATES",
                    severity,
                    f"{split}: {len(per_feature)} feature(s) have outlier candidates"
                    f"{reference}: {name_list([d['feature'] for d in per_feature])}; "
                    f"{extreme_total} extreme value(s)",
                    {
                        "split": split,
                        "z_threshold": cfg.modified_z_threshold,
                        "iqr_multiplier": cfg.iqr_multiplier,
                        "extreme_z": EXTREME_Z,
                        "features": per_feature[: cfg.top_n],
                        "feature_count": len(per_feature),
                    },
                    limitations=[
                        "max_score is |modified z| for modified_z and distance beyond the "
                        "quartiles in IQR units for iqr.",
                    ],
                )
            )

        degenerate = {
            column: {
                "n_train": s.train.n,
                "deviations_from_median_train": s.degenerate_deviations_train,
                "deviations_from_median_test": s.degenerate_deviations_test,
            }
            for column, s in screens.items()
            if s.method == "degenerate"
        }
        if degenerate:
            findings.append(
                make_finding(
                    "DEGENERATE_SCALE",
                    "info",
                    f"{len(degenerate)} feature(s) have zero MAD and zero IQR; robust outlier "
                    "screening is not meaningful for them",
                    {"features": dict(list(degenerate.items())[: cfg.top_n])},
                )
            )
        insufficient = sorted(c for c, s in screens.items() if s.method == "insufficient_data")
        skipped = (
            {"insufficient_data": f"{len(insufficient)} feature(s) with < {cfg.min_n} finite "
             f"training values: {insufficient[: cfg.top_n]}"}
            if insufficient
            else {}
        )
        metrics: dict[str, Any] = {
            "features_screened": len(screens),
            "methods": methods,
            "splits": split_metrics,
            "reference_split": "train",
        }
        return self.result(findings, metrics, skipped)
