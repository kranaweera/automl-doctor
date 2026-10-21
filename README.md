# AutoML Doctor

**Local-first statistical diagnostics for trustworthy tabular machine learning.**

AutoML Doctor (`mldoctor`) inspects your training data, held-out data and model predictions
and reports statistical problems that commonly make ML results look better, or less
certain, than they really are: missing data, imbalance, redundant features, outliers,
leakage indicators, generalization gaps, imprecise metrics and systematic errors.

- Runs entirely on your machine. No network calls, no telemetry, no LLMs, no model
  deserialization.
- Every number in a report is computed from your data. Undefined quantities are reported as
  `null` with a reason, never as made-up values.
- Findings are graded by configurable **screening heuristics**, not presented as proof. The
  tool never produces a single "model health score".
- Output is a versioned, strictly validated JSON report plus a Rich terminal view.

```text
$ mldoctor diagnose --train train.csv --test test.csv \
    --predictions test_predictions.csv --train-predictions train_predictions.csv \
    --target gait_speed --row-id visit_id --group participant_id \
    --evaluation-target unseen_groups --task regression --output report.json

 Check                    Status    Findings   Highest severity
 Missingness and data …   passed           0   —
 Class imbalance          skipped          —   —        not applicable to regression tasks
 Data leakage indicators  failed           2   critical
 Generalization gap       failed           1   high
 Performance uncertainty  warning          2   medium
 …
╭─ Data leakage indicators ───────────────────────────────────────────────────╮
│  CRITICAL  Group overlap across splits [GROUP_OVERLAP]                      │
│   6 group(s) appear in both splits (48 test rows); the declared goal is     │
│   generalization to unseen groups                                           │
│   → Use group-disjoint splitting (e.g. GroupKFold / GroupShuffleSplit) …    │
```

## Installation

Requires Python 3.11 or newer.

```bash
git clone <this repository> automl-doctor && cd automl-doctor
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .            # runtime only
pip install -e ".[dev]"     # + pytest, hypothesis, scikit-learn (tests only), ruff, mypy
mldoctor --help
```

Runtime dependencies are limited to NumPy, pandas, SciPy, Pydantic, Typer and Rich.

## Quickstart with the bundled examples

```bash
cd examples/regression
mldoctor diagnose --train train.csv --test test.csv \
  --predictions test_predictions.csv --train-predictions train_predictions.csv \
  --target price --row-id row_id --task regression --output report.json

mldoctor report report.json --show medium   # re-render a saved report
```

Three example projects live in `examples/` with documented planted issues
(see [`examples/README.md`](examples/README.md)): `regression/`, `classification/` and
`grouped_data/` (repeated measurements per participant). Each has a `sample_report.json`.

## Input contract

| Input | Required columns | Notes |
|-------|------------------|-------|
| `--train` CSV | features, target | Always required. |
| `--test` CSV | same features (strict), target | Target may be absent only if no test predictions are given. |
| `--predictions` CSV | `<row-id>`, `y_pred` | Held-out predictions. Requires `--test` and `--row-id`. |
| `--train-predictions` CSV | `<row-id>`, `y_pred` | Needed for the generalization-gap check. |

- Predictions are **joined by row ID, never by position**. Duplicate IDs, missing or extra
  prediction rows, missing evaluation targets, unknown class labels and non-finite
  regression predictions are rejected (exit code 2), not silently "fixed".
- The target, row ID, group ID and optional time column are metadata, never features.
- Train and test must have the same feature columns unless `--allow-extra-columns` is set.
- Classification uses hard labels only (`1`, `1.0` and `"1"` are treated as the same label).
  Probability inputs and calibration are out of scope for v0.1.
- IDs and group columns are read as text, so `007` stays `007`.
- Reports contain aggregate evidence only: no raw rows, IDs, group identifiers or
  individual predictions.

## Commands

| Command | Purpose |
|---------|---------|
| `mldoctor diagnose` | Run every applicable diagnostic. |
| `mldoctor data` | Dataset checks on training data only (integrity, imbalance, correlation, outliers). |
| `mldoctor leakage` | Cross-split leakage indicators. |
| `mldoctor checks` | List installed checks, supported tasks and required inputs. |
| `mldoctor report FILE` | Validate and render an existing report JSON. |
| `mldoctor version` | Print package and report-schema versions. |

Common options: `--config FILE` (JSON or TOML), `--seed`, `--evaluation-target
{unseen_rows,unseen_groups,unspecified}`, `--fail-on SEVERITY`, `--continue-on-error`,
`--allow-extra-columns`, `--output FILE` (`-` for stdout), `--quiet`, `--show SEVERITY`.

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | Run completed; no finding at or above `--fail-on`. |
| 1 | At least one finding at or above `--fail-on`. |
| 2 | Invalid CLI usage, input data or configuration. No report is written. |
| 3 | Internal error, or a diagnostic ended in `error` (the partial report is still written). `--continue-on-error` turns this into 0/1. |

Use `--fail-on high` in CI to block merges on serious findings.

## The eight diagnostics

| id | What it computes | Needs |
|----|------------------|-------|
| `data_integrity` | Missing fractions, ±Inf counts, constant/near-constant features, duplicate and discordant rows, mixed and mismatched types, missing targets | train |
| `class_imbalance` | Class counts and proportions, normalized Shannon entropy, minority count and ratio | classification |
| `feature_correlation` | Pairwise-complete Pearson and Spearman correlations with per-pair *n* | ≥ 2 numeric features |
| `outliers` | Modified z-scores (MAD), IQR fallback, degenerate-scale handling; test scored on the training reference | numeric features |
| `leakage` | Cross-split exact row overlap, group overlap, target copies, near-perfect association screen, suspicious names | train (+ test, group) |
| `overfitting` | Direction-aware train vs held-out gaps for MAE/RMSE/R² or accuracy/balanced accuracy/macro-F1 | train + test predictions |
| `uncertainty` | Percentile bootstrap CIs (B = 1000, seed 42), cluster bootstrap when `--group` is set | test predictions |
| `prediction_errors` | Residual summaries and bias, or confusion matrix and per-class precision/recall/F1 | test predictions |

Each result has a status: `passed` (applicable checks ran, no alert), `warning` (low or
medium findings), `failed` (high or critical findings), `skipped` (not applicable, with a
reason) or `error` (unexpected exception, with a message). A check that cannot run is
**never** reported as passed. Sub-checks that could not run are listed under
`skipped_subchecks`.

Full formulas, assumptions and thresholds are in
[`docs/statistical_methods.md`](docs/statistical_methods.md).

## Configuration

Precedence: built-in defaults < config file < CLI flags.

```toml
# mldoctor.toml
seed = 42
evaluation_target = "unseen_groups"   # what the held-out set is meant to measure
exclude_columns = ["free_text_notes"]

[integrity]
missing_warning_fraction = 0.05
missing_high_fraction = 0.30

[correlation]
threshold = 0.90

[uncertainty]
n_bootstrap = 2000
confidence_level = 0.95
```

See [`examples/config.example.toml`](examples/config.example.toml) for every option and its
default. The run's effective configuration is fingerprinted (SHA-256) in the report.

## Python API

```python
from mldoctor.api import diagnose
from mldoctor.reports.json_report import write_report

report = diagnose(
    train="train.csv", test="test.csv", predictions="test_predictions.csv",
    target="outcome", task="regression", row_id="row_id",
    config={"seed": 7, "evaluation_target": "unseen_rows"},
)
print(report.summary.failed, report.summary.highest_severity)
write_report(report, "report.json")
```

`diagnose` also accepts pandas DataFrames. Your frames are copied, never modified.

## Report format

Reports follow schema version `0.1.0` (see [`docs/report_schema.md`](docs/report_schema.md)).
They are validated on write and on read: there are no NaN or Infinity values, summary counts
must match the results, timestamps are timezone-aware UTC, and diagnostics and findings are
ordered deterministically. Input files are identified by SHA-256 content hashes.

## Limitations

- Leakage checks are **indicators**. Pipeline leakage (scalers, imputers or feature
  selection fit on all data, temporal look-ahead) cannot be detected from CSVs. Passing
  these checks does not show that no leakage exists.
- All thresholds are heuristics chosen for screening. They are not significance tests and
  should be tuned to your domain.
- Bootstrap intervals are conditional on the trained model and the evaluation set. They do
  not cover re-training variability, split choice or future drift, and percentile intervals
  can under-cover for small or imbalanced test sets.
- Outlier screening is univariate. Correlation screening is pairwise and misses
  multicollinearity among three or more features.
- The association screen for continuous features supports binary classification and
  regression. Continuous features with multiclass targets are not screened (this is
  reported as a skipped sub-check).
- Only hard-label classification and CSV inputs are supported. Time-series-aware checks are
  not implemented; `time_column` is treated as metadata only.

The full list, with future work, is in
[`docs/statistical_methods.md#limitations-and-future-work`](docs/statistical_methods.md#limitations-and-future-work).

## Development

```bash
pip install -e ".[dev]"
pytest                   # unit, integration, regression (snapshot) and property-based tests
ruff check src tests     # lint
mypy                     # strict type checking
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) and
[`docs/plugin_development.md`](docs/plugin_development.md) for writing new diagnostics.

## License

MIT
