"""Feature-correlation diagnostic (redundancy screening).

Numeric, non-boolean features of the training split only; target, row ID, group and
excluded columns are never included. Pearson r and Spearman rho use pairwise-complete
observations with the pair-specific ``n`` reported (see
:mod:`mldoctor.statistics.correlations`).

A pair is flagged when ``max(|r|, |rho|) ≥ threshold`` (0.90) and ``n ≥ min_pair_n`` (10):
severity ``low``, or ``medium`` when ``≥ near_perfect_threshold`` (0.99). Pairs with a
constant column on their joint rows are *undefined*, never ``r = 0``.

Flagged pairs are redundancy candidates — not evidence of leakage and not grounds for
automatic removal. For very wide data only the first ``max_features`` numeric features (in
column order) are analysed and the truncation is reported.
"""

from __future__ import annotations

from typing import Any, ClassVar

from mldoctor.core.context import DiagnosticContext
from mldoctor.core.protocols import BaseDiagnostic
from mldoctor.core.schemas import DiagnosticResult, Finding
from mldoctor.recommendations.rules import make_finding
from mldoctor.statistics.correlations import PairCorrelation, pairwise_correlations


def _pair_evidence(pair: PairCorrelation) -> dict[str, Any]:
    return {
        "feature_a": pair.a,
        "feature_b": pair.b,
        "n": pair.n,
        "pearson": None if pair.pearson is None else round(pair.pearson, 6),
        "spearman": None if pair.spearman is None else round(pair.spearman, 6),
        "spearman_exact": pair.spearman_exact,
    }


class FeatureCorrelationDiagnostic(BaseDiagnostic):
    id = "feature_correlation"
    title = "Feature correlation"
    description = "Pairwise Pearson and Spearman correlations among numeric features."
    limitations: ClassVar[tuple[str, ...]] = (
        "Only pairwise linear/monotone dependence is screened; multicollinearity involving "
        "three or more features (e.g. a = b + c) is not detected.",
        "Correlation thresholds are heuristics, not significance tests.",
    )

    def applicability(self, context: DiagnosticContext) -> str | None:
        base = super().applicability(context)
        if base is not None:
            return base
        if len(context.numeric_features) < 2:
            return f"fewer than two numeric features ({len(context.numeric_features)} found)"
        return None

    def run(self, context: DiagnosticContext) -> DiagnosticResult:
        cfg = context.config.correlation
        features = context.numeric_features
        skipped: dict[str, str] = {}
        if len(features) > cfg.max_features:
            skipped["features_beyond_cap"] = (
                f"{len(features) - cfg.max_features} numeric feature(s) beyond max_features="
                f"{cfg.max_features} were not analysed"
            )
            features = features[: cfg.max_features]

        pairs = pairwise_correlations(context.train[features], min_n=cfg.min_pair_n)
        defined = [p for p in pairs if p.max_abs is not None]
        undefined_reasons: dict[str, int] = {}
        for p in pairs:
            if p.undefined_reason:
                undefined_reasons[p.undefined_reason] = (
                    undefined_reasons.get(p.undefined_reason, 0) + 1
                )
        ranked = sorted(defined, key=lambda p: (-(p.max_abs or 0.0), p.a, p.b))

        flagged_near = [p for p in ranked if (p.max_abs or 0) >= cfg.near_perfect_threshold]
        flagged = [
            p for p in ranked if cfg.threshold <= (p.max_abs or 0) < cfg.near_perfect_threshold
        ]
        findings: list[Finding] = []
        for group, severity, label, threshold in (
            (flagged_near, "medium", "near-perfect", cfg.near_perfect_threshold),
            (flagged, "low", "high", cfg.threshold),
        ):
            if not group:
                continue
            findings.append(
                make_finding(
                    "HIGH_FEATURE_CORRELATION",
                    severity,  # type: ignore[arg-type]
                    f"{len(group)} feature pair(s) with {label} correlation "
                    f"(max(|r|, |rho|) ≥ {threshold:g})",
                    {
                        "threshold": threshold,
                        "pair_count": len(group),
                        "pairs": [_pair_evidence(p) for p in group[: cfg.top_n]],
                    },
                    limitations=["Correlation is not evidence of leakage."],
                )
            )

        constant_features = sorted(
            {
                p.a
                for p in pairs
                if p.undefined_reason == "constant_on_pair"
                and context.train[p.a].dropna().nunique() <= 1
            }
            | {
                p.b
                for p in pairs
                if p.undefined_reason == "constant_on_pair"
                and context.train[p.b].dropna().nunique() <= 1
            }
        )
        metrics: dict[str, Any] = {
            "features_analyzed": len(features),
            "pairs_total": len(pairs),
            "pairs_defined": len(defined),
            "pairs_undefined": undefined_reasons,
            "pairs_flagged": len(flagged) + len(flagged_near),
            "constant_features": constant_features,
            "min_pair_n": cfg.min_pair_n,
            "top_pairs": [_pair_evidence(p) for p in ranked[: cfg.top_n]],
        }
        return self.result(findings, metrics, skipped)
