# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-22

### Added
- `mldoctor` CLI (`diagnose`, `data`, `leakage`, `checks`, `report`, `version`) with exit
  codes 0/1/2/3, `--fail-on`, `--continue-on-error` and JSON/TOML configuration.
- Python API `mldoctor.api.diagnose` accepting CSV paths or DataFrames.
- Strict input contracts: ID-based prediction joins, duplicate/missing/unexpected ID
  rejection, strict train/test feature agreement, canonical class labels.
- Eight diagnostics: data integrity, class imbalance, feature correlation, outlier
  screening, leakage indicators, generalization gap, bootstrap uncertainty (row and
  cluster), prediction errors.
- Versioned report schema 0.1.0 with strict JSON (no NaN/Inf), deterministic ordering,
  input hashes and a config fingerprint.
- Rich terminal report.
- Example regression, classification and grouped datasets with sample reports.
- Unit, property-based (Hypothesis), integration and snapshot tests. CI on Linux, macOS and
  Windows for Python 3.11–3.13.
