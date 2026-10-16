"""Versioned finding catalog, severity rules and status policy.

Rules operate only on *computed evidence* and declared evaluation goals. Pure numerical
code lives in :mod:`mldoctor.statistics`; this module decides how that evidence is
labelled. All thresholds come from :class:`mldoctor.core.config.DoctorConfig` and are
screening heuristics, not significance tests.

Status policy (documented in ``docs/report_schema.md``):

* any ``high`` or ``critical`` finding  -> ``failed``
* otherwise any ``low`` or ``medium``   -> ``warning``
* otherwise (no findings / ``info``)    -> ``passed``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mldoctor.core.schemas import SEVERITY_ORDER, Finding, Severity, Status, jsonable

CATALOG_VERSION = "0.1.0"


@dataclass(frozen=True)
class CatalogEntry:
    title: str
    recommendation: str


CATALOG: dict[str, CatalogEntry] = {
    # --- spec-defined codes -------------------------------------------------------------
    "HIGH_MISSINGNESS": CatalogEntry(
        "High missingness",
        "Investigate why values are missing (collection process, joins, sensor failures). "
        "Missingness may be informative and is not assumed to be completely at random; "
        "choose imputation or modelling strategies deliberately and fit them on training "
        "data only.",
    ),
    "CONSTANT_FEATURE": CatalogEntry(
        "Constant feature",
        "A feature with a single observed value carries no information in this split. "
        "Confirm it is expected (e.g. a filter artefact) before deciding whether to keep it.",
    ),
    "CLASS_IMBALANCE": CatalogEntry(
        "Class imbalance",
        "Prefer metrics that remain informative under imbalance (balanced accuracy, "
        "per-class recall/precision, macro-F1) and stratified splitting. Resampling or "
        "re-weighting are modelling choices to evaluate, not automatic fixes.",
    ),
    "HIGH_FEATURE_CORRELATION": CatalogEntry(
        "Highly correlated features",
        "These pairs are redundancy candidates. Check whether one is derived from the other. "
        "Correlation is neither proof of leakage nor, by itself, grounds for removal.",
    ),
    "OUTLIER_CANDIDATES": CatalogEntry(
        "Outlier candidates",
        "Review flagged values for entry errors or unit mismatches. Legitimate rare "
        "observations can be important; do not delete them automatically.",
    ),
    "CROSS_SPLIT_DUPLICATE": CatalogEntry(
        "Feature rows shared across splits",
        "Identical feature rows in train and test can inflate held-out performance. Decide "
        "whether repeated measurements are expected for your evaluation goal; deduplicate "
        "or split by entity if the test set should contain unseen records.",
    ),
    "GROUP_OVERLAP": CatalogEntry(
        "Group overlap across splits",
        "Use group-disjoint splitting (e.g. GroupKFold / GroupShuffleSplit) if the model "
        "must generalize to unseen groups.",
    ),
    "POSSIBLE_TARGET_COPY": CatalogEntry(
        "Possible target copy",
        "A feature reproduces or deterministically encodes the target. Confirm that it "
        "would be available at prediction time; if not, remove it and re-evaluate.",
    ),
    "GENERALIZATION_GAP": CatalogEntry(
        "Generalization gap",
        "Training performance exceeds held-out performance. Possible causes include "
        "overfitting, distribution shift between splits, split errors or leakage into "
        "training. Compare against cross-validated estimates and regularized baselines.",
    ),
    "WIDE_METRIC_INTERVAL": CatalogEntry(
        "Wide metric confidence interval",
        "The held-out estimate is imprecise. Use a larger evaluation set, repeated or "
        "cross-validated evaluation, and report intervals alongside point estimates.",
    ),
    "SYSTEMATIC_BIAS": CatalogEntry(
        "Systematic prediction bias",
        "Predictions are shifted in a consistent direction. Check for target drift, "
        "mis-specified link/scale, or calibration issues; inspect errors by subgroup.",
    ),
    # --- additional codes introduced in v0.1 (documented in docs/report_schema.md) -------
    "INVALID_NUMERIC_VALUES": CatalogEntry(
        "Invalid numeric values",
        "Replace or investigate ±Infinity values at the source; most models and metrics "
        "cannot handle them and they are not ordinary missing values.",
    ),
    "NEAR_CONSTANT_FEATURE": CatalogEntry(
        "Near-constant feature",
        "Almost all rows share one value. The feature may be uninformative or unstable "
        "across splits; check whether the rare values are meaningful.",
    ),
    "DUPLICATE_ROWS": CatalogEntry(
        "Duplicate rows",
        "Confirm whether duplicates are legitimate repeated observations or ingestion "
        "artefacts. Duplicates can overweight some records and inflate evaluation.",
    ),
    "DISCORDANT_DUPLICATES": CatalogEntry(
        "Identical features with different targets",
        "Identical feature rows with different outcomes indicate irreducible noise, "
        "missing features or label errors. This is suspicious, not automatically invalid.",
    ),
    "MIXED_TYPES": CatalogEntry(
        "Mixed value types",
        "A column mixes numeric and non-numeric text. Check for placeholders such as "
        "'n/a' or '?', unit suffixes or locale formatting.",
    ),
    "DTYPE_MISMATCH": CatalogEntry(
        "Type mismatch across splits",
        "The same column is parsed with different types in train and test. Align parsing "
        "and encoding before modelling.",
    ),
    "MISSING_TARGET": CatalogEntry(
        "Missing target values",
        "Rows without a target cannot be used for supervised training or evaluation. "
        "Investigate why labels are missing; it may be informative.",
    ),
    "ONE_CLASS_TARGET": CatalogEntry(
        "Single-class target",
        "Only one class is present, so most classification metrics are undefined or "
        "degenerate. Verify the split and labelling.",
    ),
    "CLASS_MISSING_IN_SPLIT": CatalogEntry(
        "Class absent from a split",
        "Some classes appear in only one split. Per-class metrics for these classes are "
        "undefined; consider stratified splitting.",
    ),
    "DEGENERATE_SCALE": CatalogEntry(
        "Degenerate scale",
        "Both MAD and IQR are zero, so robust outlier screening is not statistically "
        "meaningful. Values different from the median are counted but not called outliers.",
    ),
    "SUSPICIOUS_TARGET_ASSOCIATION": CatalogEntry(
        "Near-perfect feature–target association",
        "This feature almost determines the target in training data. That can be a "
        "legitimate strong predictor or a leak; confirm it is available at prediction time.",
    ),
    "SUSPICIOUS_FEATURE_NAME": CatalogEntry(
        "Suspicious feature name",
        "The feature name resembles the target or a post-outcome quantity. Confirm it is "
        "measured before the prediction time.",
    ),
    "UNDEFINED_METRIC": CatalogEntry(
        "Undefined metric",
        "A metric could not be computed for this data (see reason). Use metrics that are "
        "defined for the available classes or values.",
    ),
    "INSUFFICIENT_VALID_REPLICATES": CatalogEntry(
        "Too few valid bootstrap replicates",
        "Many bootstrap resamples produced undefined metrics, so the interval is unreliable. "
        "Increase the evaluation set or use a metric defined for all resamples.",
    ),
    "SMALL_EVALUATION_SET": CatalogEntry(
        "Small evaluation set",
        "Percentile bootstrap intervals can be unstable with few independent resampling "
        "units. Treat intervals as rough and prefer larger or repeated evaluation.",
    ),
    "CLASS_NEVER_PREDICTED": CatalogEntry(
        "Class never predicted",
        "The model never predicts some classes that occur in the evaluation data; their "
        "recall is zero. Review class weighting, thresholds and feature coverage.",
    ),
    "PREDICTION_COLLAPSE": CatalogEntry(
        "Prediction collapse",
        "The model predicts a single class for every row. Check training, class weighting "
        "and whether predictions were exported correctly.",
    ),
    "TEST_OUTPERFORMS_TRAIN": CatalogEntry(
        "Held-out performance exceeds training",
        "Better held-out than training performance is unusual. Possible causes include "
        "distribution differences between splits, a small or easy test set, heavy "
        "regularization, or mislabelled split files.",
    ),
}


def make_finding(
    code: str,
    severity: Severity,
    message: str,
    evidence: dict[str, Any],
    *,
    recommendation: str | None = None,
    limitations: list[str] | None = None,
    title: str | None = None,
) -> Finding:
    """Create a :class:`Finding` from the catalog, converting evidence to JSON types."""
    entry = CATALOG.get(code)
    if entry is None:
        raise KeyError(f"unknown finding code '{code}'; add it to the catalog")
    return Finding(
        code=code,
        severity=severity,
        title=title or entry.title,
        message=message,
        evidence=jsonable(evidence),
        recommendation=recommendation or entry.recommendation,
        limitations=limitations or [],
    )


def status_from_findings(findings: list[Finding]) -> Status:
    if any(SEVERITY_ORDER[f.severity] >= SEVERITY_ORDER["high"] for f in findings):
        return "failed"
    if any(SEVERITY_ORDER[f.severity] >= SEVERITY_ORDER["low"] for f in findings):
        return "warning"
    return "passed"


def severity_at_least(severity: Severity, threshold: Severity) -> bool:
    return SEVERITY_ORDER[severity] >= SEVERITY_ORDER[threshold]


def sort_findings(findings: list[Finding]) -> list[Finding]:
    """Deterministic order: most severe first, then code, then message."""
    return sorted(findings, key=lambda f: (-SEVERITY_ORDER[f.severity], f.code, f.message))
