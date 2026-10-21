"""Held-out performance uncertainty via the nonparametric bootstrap.

``B = 1000`` replicates (configurable) drawn from a seeded, per-diagnostic random stream.
When a group column is declared, whole groups are resampled (cluster bootstrap) because
rows from the same group are not independent; otherwise rows are resampled. The same
replicate index sets are reused for every metric.

Findings:

* ``WIDE_METRIC_INTERVAL`` – CI width > 0.10 for bounded scores (accuracy, balanced
  accuracy, macro-F1, R²) or > 50% of the estimate for losses (MAE, RMSE). ``medium`` if
  more than twice the threshold, else ``low``.
* ``INSUFFICIENT_VALID_REPLICATES`` – fewer than 80% of replicates gave a defined value.
* ``SMALL_EVALUATION_SET`` – fewer than 30 independent resampling units.

The interval is conditional on the evaluation set and the trained model; it is not
uncertainty about re-training or future drift.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from mldoctor.core.context import DiagnosticContext
from mldoctor.core.protocols import BaseDiagnostic, Requirement
from mldoctor.core.schemas import DiagnosticResult, Finding, Severity
from mldoctor.recommendations.rules import make_finding
from mldoctor.statistics.bootstrap import (
    BootstrapResult,
    Resampler,
    bootstrap_statistic,
    classification_statistics,
    group_resampler,
    regression_statistics,
    row_resampler,
)
from mldoctor.statistics.metrics import fmt_number, sig

BOUNDED = {"accuracy", "balanced_accuracy", "macro_f1", "r2"}


def _opt(value: float | None) -> float | None:
    return None if value is None else sig(value)


class UncertaintyDiagnostic(BaseDiagnostic):
    id = "uncertainty"
    title = "Performance uncertainty"
    description = "Percentile bootstrap confidence intervals (row or group resampling)."
    requires: ClassVar[frozenset[Requirement]] = frozenset({"test_predictions", "test_target"})
    limitations: ClassVar[tuple[str, ...]] = (
        "Conditional on this evaluation set and trained model: does not reflect re-training "
        "variability, alternative splits or future temporal drift.",
        "Percentile intervals can under-cover for small or imbalanced evaluation sets.",
        "Replicates with undefined metrics are dropped; the interval is conditional on the "
        "valid replicates.",
    )

    def run(self, context: DiagnosticContext) -> DiagnosticResult:
        assert context.test is not None and context.test_predictions is not None
        cfg = context.config.uncertainty
        test = context.test
        rng = context.rng(self.id)
        resampler: Resampler
        if context.group is not None:
            resampler = group_resampler(test[context.group].to_numpy(), cfg.n_bootstrap, rng)
        else:
            resampler = row_resampler(len(test), cfg.n_bootstrap, rng)

        if context.task == "regression":
            stats = regression_statistics(
                test[context.target].to_numpy(dtype=float),
                context.test_predictions.to_numpy(dtype=float),
            )
        else:
            stats = classification_statistics(
                test[context.target].astype(str).tolist(),
                context.test_predictions.astype(str).tolist(),
                context.class_labels or [],
            )

        results: dict[str, BootstrapResult] = {
            name: bootstrap_statistic(fn, resampler, confidence=cfg.confidence_level)
            for name, fn in stats.items()
        }
        findings: list[Finding] = []
        wide: list[dict[str, Any]] = []
        wide_severity: Severity = "low"
        low_valid: dict[str, Any] = {}
        payload: dict[str, Any] = {}
        for name, res in results.items():
            payload[name] = {
                "estimate": _opt(res.estimate),
                "ci_low": _opt(res.ci_low),
                "ci_high": _opt(res.ci_high),
                "width": _opt(res.width),
                "replicate_sd": _opt(res.replicate_sd),
                "n_valid_replicates": res.n_valid,
                "valid_fraction": sig(res.valid_fraction),
                "undefined_reason": res.estimate_reason or res.ci_reason,
            }
            if res.estimate is not None and res.valid_fraction < cfg.min_valid_fraction:
                low_valid[name] = {"n_valid": res.n_valid, "n_replicates": res.n_replicates}
            width = res.width
            if width is None or res.estimate is None:
                continue
            if name in BOUNDED:
                measure, threshold = width, cfg.bounded_width_threshold
            else:
                if res.estimate == 0:
                    continue
                measure, threshold = width / abs(res.estimate), cfg.relative_width_threshold
            if measure > threshold:
                wide.append({"metric": name, "estimate": sig(res.estimate),
                             "ci_low": _opt(res.ci_low), "ci_high": _opt(res.ci_high),
                             "width": sig(width),
                             "measure": "absolute_width" if name in BOUNDED else "relative_width",
                             "value": sig(measure), "threshold": threshold})
                if measure > 2 * threshold:
                    wide_severity = "medium"

        unit_desc = (
            f"{resampler.n_units} groups ('{context.group}')"
            if resampler.unit == "group" else f"{resampler.n_units} rows"
        )
        if wide:
            findings.append(
                make_finding(
                    "WIDE_METRIC_INTERVAL",
                    wide_severity,
                    f"{len(wide)} metric(s) have wide {cfg.confidence_level:.0%} intervals "
                    f"(resampling {unit_desc}): "
                    + ", ".join(
                        f"{w['metric']} {fmt_number(w['estimate'])} "
                        f"[{fmt_number(w['ci_low'])}, {fmt_number(w['ci_high'])}]"
                        for w in wide
                    ),
                    {"metrics": wide, "resampling_unit": resampler.unit,
                     "n_units": resampler.n_units},
                )
            )
        if low_valid:
            findings.append(
                make_finding(
                    "INSUFFICIENT_VALID_REPLICATES",
                    "medium",
                    f"{len(low_valid)} metric(s) were undefined on more than "
                    f"{1 - cfg.min_valid_fraction:.0%} of bootstrap replicates",
                    {"metrics": low_valid, "min_valid_fraction": cfg.min_valid_fraction},
                )
            )
        if resampler.n_units < cfg.min_resampling_units:
            findings.append(
                make_finding(
                    "SMALL_EVALUATION_SET",
                    "low",
                    f"only {unit_desc} available as independent resampling units "
                    f"(< {cfg.min_resampling_units})",
                    {"n_units": resampler.n_units, "resampling_unit": resampler.unit,
                     "min_resampling_units": cfg.min_resampling_units},
                )
            )

        group_sizes = None
        if resampler.unit == "group":
            sizes = np.bincount(
                np.unique(test[context.group].to_numpy(), return_inverse=True)[1]
            )
            group_sizes = {"min": int(sizes.min()), "median": sig(float(np.median(sizes))),
                           "max": int(sizes.max())}
        metrics: dict[str, Any] = {
            "estimand": (
                "performance of the fixed trained model on the population represented by the "
                "evaluation set"
            ),
            "method": "percentile bootstrap",
            "resampling_unit": resampler.unit,
            "n_units": resampler.n_units,
            "n_rows": resampler.n_rows,
            "group_sizes": group_sizes,
            "n_bootstrap": cfg.n_bootstrap,
            "confidence_level": cfg.confidence_level,
            "seed": context.seed,
            "random_stream": self.id,
            "intervals": payload,
        }
        return self.result(findings, metrics)
