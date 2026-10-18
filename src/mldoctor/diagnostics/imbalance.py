"""Class-imbalance diagnostic (classification only).

For each split with labels: per-class counts ``n_k``, proportions ``p_k``, the
majority/minority ratio and normalized Shannon entropy ``H = -Σ p_k ln p_k / ln K``
(``K`` = number of declared classes; see :func:`mldoctor.statistics.metrics.class_distribution`).

Alert rule (configurable heuristics): minority count < ``min_minority_count`` (20) OR
minority proportion < ``min_minority_proportion`` (0.05). Severity is ``high`` when the
minority count is below ``severe_minority_count`` (10) or its proportion below
``severe_minority_proportion`` (0.02), otherwise ``medium``.

Imbalance is a property of the data, not a defect. The report explains metric implications
and never prescribes resampling automatically.
"""

from __future__ import annotations

from typing import Any, ClassVar

from mldoctor.core.context import DiagnosticContext
from mldoctor.core.protocols import BaseDiagnostic
from mldoctor.core.schemas import DiagnosticResult, Finding, Severity, Task
from mldoctor.recommendations.rules import make_finding
from mldoctor.statistics.metrics import class_distribution


class ClassImbalanceDiagnostic(BaseDiagnostic):
    id = "class_imbalance"
    title = "Class imbalance"
    description = "Class frequencies, normalized entropy, minority-class counts per split."
    supported_tasks: ClassVar[frozenset[Task]] = frozenset({"classification"})
    limitations: ClassVar[tuple[str, ...]] = (
        "Thresholds are contextual heuristics; acceptable imbalance depends on the "
        "application and the metric used.",
    )

    def run(self, context: DiagnosticContext) -> DiagnosticResult:
        cfg = context.config.imbalance
        labels = context.class_labels or []
        findings: list[Finding] = []
        metrics: dict[str, Any] = {"declared_classes": labels, "splits": {}}
        skipped: dict[str, str] = {}

        splits = [("train", context.train)]
        if context.test is not None:
            if context.test_has_target:
                splits.append(("test", context.test))
            else:
                skipped["test"] = "test split has no target column"

        for split, frame in splits:
            dist = class_distribution(frame[context.target].tolist(), labels)
            minority_prop = dist.minority_proportion
            metrics["splits"][split] = {
                "n_labelled": dist.n,
                "counts": dict(zip(dist.labels, dist.counts, strict=True)),
                "proportions": {
                    k: round(v, 6) for k, v in zip(dist.labels, dist.proportions, strict=True)
                },
                "observed_classes": dist.observed_classes,
                "normalized_entropy": (
                    None if dist.entropy_normalized is None else round(dist.entropy_normalized, 6)
                ),
                "normalized_entropy_undefined_reason": dist.entropy_undefined_reason,
                "majority_count": dist.majority_count,
                "minority_count": dist.minority_count,
                "minority_proportion": None if minority_prop is None else round(minority_prop, 6),
                "imbalance_ratio": (
                    None if dist.imbalance_ratio is None else round(dist.imbalance_ratio, 6)
                ),
                "imbalance_ratio_undefined_reason": dist.imbalance_ratio_undefined_reason,
            }
            if dist.n == 0:
                continue

            if dist.observed_classes <= 1:
                findings.append(
                    make_finding(
                        "ONE_CLASS_TARGET",
                        "high",
                        f"{split}: only one class is present among {dist.n} labelled rows",
                        {"split": split, "observed_classes": dist.observed_classes,
                         "declared_classes": len(labels)},
                    )
                )
                continue

            if dist.absent_labels:
                findings.append(
                    make_finding(
                        "CLASS_MISSING_IN_SPLIT",
                        "medium",
                        f"{split}: {len(dist.absent_labels)} declared class(es) have no examples",
                        {"split": split, "absent_classes": dist.absent_labels},
                    )
                )

            # Imbalance among the classes actually present in this split.
            present_counts = [c for c in dist.counts if c > 0]
            minority = min(present_counts)
            minority_label = next(
                lbl for lbl, c in zip(dist.labels, dist.counts, strict=True) if c == minority
            )
            proportion = minority / dist.n
            triggered = []
            if minority < cfg.min_minority_count:
                triggered.append(f"minority count {minority} < {cfg.min_minority_count}")
            if proportion < cfg.min_minority_proportion:
                triggered.append(
                    f"minority proportion {proportion:.4f} < {cfg.min_minority_proportion}"
                )
            if not triggered:
                continue
            severe = (
                minority < cfg.severe_minority_count
                or proportion < cfg.severe_minority_proportion
            )
            severity: Severity = "high" if severe else "medium"
            findings.append(
                make_finding(
                    "CLASS_IMBALANCE",
                    severity,
                    f"{split}: smallest class '{minority_label}' has {minority} of {dist.n} rows "
                    f"({proportion:.2%}); " + "; ".join(triggered),
                    {
                        "split": split,
                        "minority_class": minority_label,
                        "minority_count": minority,
                        "minority_proportion": round(proportion, 6),
                        "majority_count": max(present_counts),
                        "imbalance_ratio_present_classes": round(max(present_counts) / minority, 6),
                        "normalized_entropy": metrics["splits"][split]["normalized_entropy"],
                        "thresholds": {
                            "min_minority_count": cfg.min_minority_count,
                            "min_minority_proportion": cfg.min_minority_proportion,
                            "severe_minority_count": cfg.severe_minority_count,
                            "severe_minority_proportion": cfg.severe_minority_proportion,
                        },
                    },
                    limitations=[
                        "Accuracy can look high by predicting the majority class; per-class "
                        "metrics for the minority class have high variance."
                    ],
                )
            )
        return self.result(findings, metrics, skipped)
