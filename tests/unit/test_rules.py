from __future__ import annotations

import re
from pathlib import Path

import pytest

from mldoctor.recommendations.rules import (
    CATALOG,
    make_finding,
    severity_at_least,
    sort_findings,
    status_from_findings,
)

SPEC_CODES = {
    "HIGH_MISSINGNESS", "CONSTANT_FEATURE", "CLASS_IMBALANCE", "HIGH_FEATURE_CORRELATION",
    "OUTLIER_CANDIDATES", "CROSS_SPLIT_DUPLICATE", "GROUP_OVERLAP", "POSSIBLE_TARGET_COPY",
    "GENERALIZATION_GAP", "WIDE_METRIC_INTERVAL", "SYSTEMATIC_BIAS",
}


def test_catalog_contains_all_spec_codes() -> None:
    assert set(CATALOG) >= SPEC_CODES


def test_every_code_used_in_source_is_catalogued() -> None:
    src = Path(__file__).resolve().parents[2] / "src" / "mldoctor" / "diagnostics"
    used = set()
    for path in src.glob("*.py"):
        used |= set(re.findall(r'make_finding\(\s*"([A-Z_]+)"', path.read_text()))
    assert used, "expected to find make_finding calls"
    assert used <= set(CATALOG)


@pytest.mark.parametrize(
    ("severities", "status"),
    [([], "passed"), (["info"], "passed"), (["info", "low"], "warning"),
     (["medium"], "warning"), (["low", "high"], "failed"), (["critical"], "failed")],
)
def test_status_policy(severities: list[str], status: str) -> None:
    findings = [make_finding("CONSTANT_FEATURE", s, "m", {}) for s in severities]  # type: ignore[arg-type]
    assert status_from_findings(findings) == status


def test_unknown_code_rejected() -> None:
    with pytest.raises(KeyError):
        make_finding("NOT_A_CODE", "low", "m", {})


def test_sorting_and_threshold_helpers() -> None:
    fs = [make_finding("CONSTANT_FEATURE", "low", "b", {}),
          make_finding("GROUP_OVERLAP", "critical", "a", {}),
          make_finding("CLASS_IMBALANCE", "low", "a", {})]
    assert [f.code for f in sort_findings(fs)] == [
        "GROUP_OVERLAP", "CLASS_IMBALANCE", "CONSTANT_FEATURE"]
    assert severity_at_least("high", "medium") and not severity_at_least("low", "medium")
