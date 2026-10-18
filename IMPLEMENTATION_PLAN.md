# AutoML Doctor v0.1 — Implementation Plan

Source of requirements: `AutoML_Doctor_v0.1_Agent_Implementation_Spec.docx` (the "spec").
Where the spec is silent, decisions are recorded in [Decisions](#decisions-and-deviations).

## Status overview

| Milestone | Scope | Status |
|-----------|-------|--------|
| 1 — Foundation | packaging, schemas, config, context, protocols, registry, engine, CSV/prediction loaders, validation, CLI skeleton, JSON export | complete (2026-10-17) |
| 2 — Data checks | data integrity, class imbalance, feature correlation, outliers, pure statistics | complete (2026-10-18) |
| 3 — ML checks | leakage indicators, prediction errors, generalization gap | not started |
| 4 — Uncertainty | row/group bootstrap, severity rules, recommendation catalog | not started |
| 5 — Release | Rich terminal report, docs, examples, integration tests, CI, contributor docs, changelog | not started |

## Architecture

```
src/mldoctor/
  cli.py                     Typer CLI (diagnose, data, leakage, checks, report, version)
  api.py                     Python API: diagnose(...) -> Report
  core/config.py             Pydantic config (defaults < config file < CLI)
  core/context.py            DiagnosticContext shared by all checks (frames, aligned predictions, seeded RNG)
  core/protocols.py          Diagnostic protocol + BaseDiagnostic helper
  core/registry.py           Ordered registry of diagnostics (+ opt-in entry-point plugins)
  core/engine.py             Runs diagnostics, isolates exceptions into `error` results, builds Report
  core/schemas.py            Finding / DiagnosticResult / Report (strict JSON, no NaN/Inf)
  io/csv_loader.py           Local CSV loading + content hashing
  io/prediction_loader.py    Prediction CSV loading and ID-based alignment
  io/validation.py           Input contract validation -> InputValidationError (exit code 2)
  statistics/robust.py       median / MAD / modified z / IQR fences
  statistics/correlations.py pairwise-complete Pearson & Spearman
  statistics/metrics.py      regression + classification metrics with explicit undefined handling
  statistics/bootstrap.py    row and cluster (group) bootstrap, percentile CI
  diagnostics/*.py           the eight diagnostics (compute -> interpret via rules)
  recommendations/rules.py   versioned finding-code catalog + severity rules + status policy
  reports/json_report.py     versioned JSON write/read/validate
  reports/terminal_report.py Rich rendering
```

Pure numerical code (`statistics/`) never decides severity. Diagnostics call the statistics
layer, then hand numeric evidence to `recommendations/rules.py`, which maps evidence to
severities using configured thresholds.

## Milestone plan and acceptance gates

### Milestone 1 — Foundation
- [x] `pyproject.toml`, src layout, `mldoctor` console script, `py.typed`
- [x] Pydantic schemas with strict JSON validation (no NaN/Inf, counts consistent, enums, ISO timestamps)
- [x] Config model with per-diagnostic thresholds, config file (JSON/TOML) loading, fingerprint
- [x] DiagnosticContext with seeded per-diagnostic RNG streams
- [x] Diagnostic protocol, registry, engine (exceptions -> `error` results)
- [x] CSV loader, prediction loader (join by row ID), validation errors
- [x] CLI skeleton with exit codes 0/1/2/3, JSON export
- Gate: install, validate, produce a correct empty/skip JSON report.
- Result: `pip install -e .[dev]` OK; `mldoctor diagnose` on a 4-row CSV writes a schema-valid report
  with 0 checks; invalid target -> exit 2. pytest 46 passed; ruff clean; mypy --strict clean.

### Milestone 2 — Dataset diagnostics
- [x] statistics/robust.py, statistics/correlations.py
- [x] data_integrity, imbalance, correlation, outliers
- Gate: synthetic datasets produce expected values/warnings.
- Result: 40% missing -> HIGH_MISSINGNESS(high) on the right split; 990:10 -> CLASS_IMBALANCE(high)
  with exact entropy; B=2A -> r=1 flagged; MAD=0 -> IQR fallback / degenerate without division;
  correlations match scipy with missing data. pytest 84 passed; ruff + mypy clean.

### Milestone 3 — Model diagnostics
- [ ] statistics/metrics.py
- [ ] leakage, prediction_errors, overfitting
- Gate: known group overlap and model failure fixtures detected.

### Milestone 4 — Statistical reliability
- [ ] statistics/bootstrap.py (row + group), uncertainty diagnostic
- [ ] rules catalog, missing-evidence policies
- Gate: seeded intervals reproducible; small samples handled.

### Milestone 5 — Release readiness
- [ ] Rich terminal report, README, docs/, examples + generator, sample JSON
- [ ] integration + regression tests, CI workflow, CONTRIBUTING, CHANGELOG, templates
- Gate: end-to-end reproducible demo; schema and test suites pass.

## Decisions and deviations

Recorded as they are made; see final section of this file once complete.

## Commands

```bash
python3.11 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
.venv/bin/ruff check src tests
.venv/bin/mypy
```
