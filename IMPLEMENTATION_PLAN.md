# AutoML Doctor v0.1 — Implementation Plan

Source of requirements: `AutoML_Doctor_v0.1_Agent_Implementation_Spec.docx` (the "spec").
Where the spec is silent, decisions are recorded in [Decisions](#decisions-and-deviations).

## Status overview

| Milestone | Scope | Status |
|-----------|-------|--------|
| 1 — Foundation | packaging, schemas, config, context, protocols, registry, engine, CSV/prediction loaders, validation, CLI skeleton, JSON export | complete (2026-10-17) |
| 2 — Data checks | data integrity, class imbalance, feature correlation, outliers, pure statistics | complete (2026-10-18) |
| 3 — ML checks | leakage indicators, prediction errors, generalization gap | complete (2026-10-20) |
| 4 — Uncertainty | row/group bootstrap, severity rules, recommendation catalog | complete (2026-10-21) |
| 5 — Release | Rich terminal report, docs, examples, integration tests, CI, contributor docs, changelog | complete (2026-10-22) |

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
- [x] statistics/metrics.py
- [x] leakage, prediction_errors, overfitting
- Gate: known group overlap and model failure fixtures detected.
- Result: group overlap critical/medium/info by declared goal; R² .95→.20 flagged high with
  gap .75; shuffled prediction CSV gives identical metrics; metrics match scikit-learn
  (property test); one-class test set handled. pytest 114 passed; ruff + mypy clean.

### Milestone 4 — Statistical reliability
- [x] statistics/bootstrap.py (row + group), uncertainty diagnostic
- [x] rules catalog, missing-evidence policies
- Gate: seeded intervals reproducible; small samples handled.
- Result: same seed -> identical intervals, different seed -> different; cluster bootstrap
  draws whole groups with multiplicity and is ~3x wider than row bootstrap under strong
  clustering; MAE CI matches normal approximation; undefined replicates counted (valid
  fraction ≈ 1-(19/20)^20). Uncertainty results independent of which other checks run.

### Milestone 5 — Release readiness
- [x] Rich terminal report, README, docs/, examples + generator, sample JSON
- [x] integration + regression tests, CI workflow, CONTRIBUTING, CHANGELOG, templates
- Gate: end-to-end reproducible demo; schema and test suites pass.
- Result: three example projects with sample reports; CLI end-to-end tests (exit codes
  0/1/2/3, data-only, leakage, stdout, config precedence, partial report on error); golden
  snapshots; 158 tests pass on Python 3.11 and 3.13 (macOS); line coverage 94%; ruff and
  mypy --strict clean. The CI workflow (Linux/macOS/Windows × 3.11–3.13) is written but
  **has not run yet** (no remote).

## Decisions and deviations

These are decisions where the spec was silent, plus deliberate deviations. None changes a
required spec interface; all additions are backwards compatible.

### Interfaces and schema
1. **Schema extensions** (additive): `DiagnosticResult` gains `title`, `version`,
   `skipped_subchecks`, `limitations`, `error_message` and `duration_ms`. `Finding` gains
   `limitations`. `run`, `dataset` and `summary` gain provenance fields. See
   `docs/report_schema.md`.
2. **Additional finding codes** beyond the spec's 11 (e.g. `DISCORDANT_DUPLICATES`,
   `PREDICTION_COLLAPSE`, `UNEVEN_CLASS_PERFORMANCE`). All are in the catalog and documented.
3. **Diagnostic ids**: `data_integrity`, `class_imbalance`, `feature_correlation`,
   `outliers`, `leakage`, `overfitting`, `uncertainty`, `prediction_errors`, in spec order.
4. **Evaluation target** has a third value, `unspecified` (the default). Group overlap is
   then `medium` with a prompt to declare the goal, instead of guessing `critical` or `info`.
5. **Partial checks.** When some sub-checks of a diagnostic cannot run (e.g. leakage without
   a test split), the diagnostic reports `passed`/`warning`/`failed` for what ran and lists
   the rest in `skipped_subchecks`. It is not marked `skipped` as a whole.
6. **Config formats**: JSON and TOML (stdlib `tomllib`). YAML is not supported, to keep
   dependencies small.
7. **scikit-learn** is a dev/test dependency only, used as a reference implementation.
8. **Plugins** in the `mldoctor.diagnostics` entry-point group load only with
   `load_plugins = true`, because loading runs third-party code.
9. **CLI extras**: `--evaluation-target`, `--fail-on`, `--continue-on-error`,
   `--allow-extra-columns`, `--quiet`, `--show` and `--output -` (stdout). `time_column` is
   config-only and treated as metadata.

### Input contract
10. Group IDs must be complete (missing → input error), because group-aware checks cannot
    place such rows.
11. The test split may omit the target only when no test predictions are supplied (this
    gives a warning).
12. Row-ID uniqueness is enforced only where predictions are joined, as in the spec;
    elsewhere it gives a warning.
13. Classification labels are canonicalised (`1`, `1.0` and `"1"` all become `"1"`). Predicted
    labels must be among the labels observed in train ∪ test.
14. Booleans are categorical and are excluded from correlation and outlier numerics.

### Statistical choices (all thresholds configurable)
15. Missingness: medium ≥ 0.05, high ≥ 0.30 (spec). Near-constant ≥ 99% (info).
16. Imbalance: alert at minority count < 20 or proportion < 0.05 (spec). **High** below 10 or
    0.02 is our choice. Entropy uses K = number of declared classes.
17. Correlation: flag max(|r|, |ρ|) ≥ 0.90 (low) or ≥ 0.99 (medium), with n ≥ 10 (spec).
18. Outliers: |z| > 3.5 (spec). An extreme value (|z| > 10, or > 3 IQR beyond the quartiles)
    gives medium; ≥ 1% candidates gives low; otherwise info.
19. Cross-split duplicates: high at ≥ 10% of test rows. Downgraded to low when the feature
    space is low-cardinality (distinct training rows < 50% of rows).
20. Association screen (medium, warning-only): |r| or |ρ| ≥ 0.98, η ≥ 0.98, AUC ≥ 0.99,
    purity ≥ 0.99. Continuous features with multiclass targets are not screened; this is
    reported as a skipped sub-check.
21. Generalization-gap thresholds: scores 0.05/0.15, R² 0.10/0.25, losses 25%/50% relative.
22. Bootstrap: percentile CI, B = 1000, seed 42, minimum valid fraction 0.8 (spec). A
    replicate missing a class present in the full set is invalid for macro metrics. The
    cluster bootstrap is used whenever a group column is declared. Wide-interval thresholds
    are 0.10 (bounded metrics) and 50% relative (losses).
23. Balanced accuracy is **undefined** with fewer than 2 classes present (scikit-learn returns
    the single recall). Macro-F1 averages over y ∪ ŷ (scikit-learn's default).
24. Regression systematic bias uses the scale-free bias fraction mean(e)² / mean(e²) ≥ 0.25.
25. Reported floats use 10 significant digits rather than fixed decimals, so small-scale
    targets keep their precision.

## Known limitations

See `docs/statistical_methods.md#limitations-and-future-work` and the README. The main ones:

- Leakage checks cannot detect pipeline leakage.
- Thresholds are heuristics.
- Bootstrap CIs are conditional on the fitted model and the evaluation set.
- Outlier screening is univariate, and correlation screening is pairwise.
- Classification supports hard labels only, input is CSV only, and there are no time-aware
  checks.

## Next development priorities

1. Run CI on GitHub (Linux/macOS/Windows) and fix any platform-specific issues.
2. Probability inputs: calibration (reliability curves, Brier score), ROC/PR AUC with
   intervals.
3. A covariate-shift diagnostic (train vs test), e.g. a classifier two-sample test.
4. Time-aware leakage checks using `time_column`.
5. BCa intervals, and a stratified bootstrap for rare classes.
6. A multicollinearity (VIF) screen and a continuous × multiclass association screen.
7. Benchmarks on wide and large data (no performance claims are made yet).

## Commands

```bash
python3.11 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest                        # 158 tests
.venv/bin/pytest --cov=mldoctor         # coverage
.venv/bin/ruff check src tests examples
.venv/bin/mypy
PATH="$PWD/.venv/bin:$PATH" examples/run_examples.sh   # regenerate examples + sample reports
UPDATE_SNAPSHOTS=1 .venv/bin/pytest tests/regression   # after intended output changes
```
