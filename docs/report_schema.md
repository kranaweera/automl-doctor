# Report schema (version 0.1.0)

Reports are Pydantic models (`mldoctor.core.schemas.Report`) serialized as strict JSON. They
are validated when written and when read (`mldoctor report`, `load_report`).

## Guarantees

- **No NaN or Infinity.** Non-finite numbers are rejected by schema validation and by
  `json.dumps(allow_nan=False)`. Undefined quantities are `null`, with a reason in a nearby
  `*_undefined_reason` key, an `undefined` mapping or `skip_reason`.
- **Consistent summary.** `summary` counts must equal the statuses in `diagnostics`, and
  `summary.highest_severity` must equal the most severe finding.
- **Enums.** `status ∈ {passed, warning, failed, skipped, error}`,
  `severity ∈ {info, low, medium, high, critical}`, `task ∈ {classification, regression}`.
- **Status invariants.** `skipped` requires `skip_reason` and no findings. `error` requires
  `error_message` and no findings. Other statuses must not set `skip_reason`.
- **Deterministic ordering.** Diagnostics appear in registry order (`data_integrity`,
  `class_imbalance`, `feature_correlation`, `outliers`, `leakage`, `overfitting`,
  `uncertainty`, `prediction_errors`). Findings are sorted by severity (descending), then
  code, then message.
- **Timestamps** are ISO-8601, timezone-aware UTC.
- **Privacy.** No raw rows, row IDs, group identifiers or individual predictions. Feature
  and class names do appear.
- **Provenance.** `dataset.inputs[].sha256` holds content hashes of input files (null for
  in-memory DataFrames). `run.config_fingerprint` is the SHA-256 of the canonical effective
  configuration.

## Structure

```jsonc
{
  "schema_version": "0.1.0",
  "tool": {"name": "automl-doctor", "version": "0.1.0"},
  "run": {
    "id": "uuid4", "task": "regression", "seed": 42,
    "started_at": "2026-10-22T09:00:00Z", "completed_at": "2026-10-22T09:00:01Z",
    "command": "diagnose", "config_fingerprint": "sha256…",
    "evaluation_target": "unseen_groups"
  },
  "dataset": {
    "train_rows": 360, "test_rows": 168, "feature_count": 2, "numeric_feature_count": 2,
    "target": "gait_speed", "row_id_column": "visit_id", "group_column": "participant_id",
    "has_test_predictions": true, "has_train_predictions": true, "class_labels": null,
    "inputs": [{"role": "train", "sha256": "…", "rows": 360, "columns": 5}]
  },
  "summary": {
    "checks_total": 8, "passed": 4, "warning": 1, "failed": 2, "skipped": 1, "error": 0,
    "highest_severity": "critical",
    "findings_by_severity": {"info": 2, "low": 1, "medium": 1, "high": 2, "critical": 1}
  },
  "diagnostics": [
    {
      "id": "leakage", "title": "Data leakage indicators", "version": "0.1.0",
      "status": "failed",
      "findings": [{
        "code": "GROUP_OVERLAP", "severity": "critical",
        "title": "Group overlap across splits",
        "message": "6 group(s) appear in both splits (48 test rows); …",
        "evidence": {"overlapping_groups": 6, "test_rows_in_overlapping_groups": 48, "…": "…"},
        "recommendation": "Use group-disjoint splitting …",
        "limitations": ["Group identifiers themselves are not reported."]
      }],
      "metrics": {"group_overlap": {"…": "…"}, "evaluation_target": "unseen_groups"},
      "skip_reason": null,
      "skipped_subchecks": {},
      "limitations": ["Pipeline/preprocessing leakage … cannot be detected …"],
      "error_message": null,
      "duration_ms": 3.1
    }
  ],
  "execution": {"duration_ms": 245.0, "warnings": []}
}
```

### Extensions to the minimal specification schema

The specification's `DiagnosticResult` has `id, status, findings, metrics, skip_reason`.
v0.1 adds these optional fields, all backwards compatible:

| Field | Purpose |
|-------|---------|
| `title`, `version` | Human-readable name and diagnostic version |
| `skipped_subchecks` | `{name: reason}` for parts of an applicable check that could not run (prevents false passes) |
| `limitations` | Method-level caveats attached to every result |
| `error_message` | Exception type, message and location for `error` results |
| `duration_ms` | Wall-clock time of the check |

`Finding` adds `limitations: list[str]`. `run` adds `started_at`, `command`,
`config_fingerprint` and `evaluation_target`. `dataset` adds column names, prediction
availability, class labels and input hashes. `summary` adds `findings_by_severity`.

## Finding codes (catalog version 0.1.0)

Spec codes: `HIGH_MISSINGNESS`, `CONSTANT_FEATURE`, `CLASS_IMBALANCE`,
`HIGH_FEATURE_CORRELATION`, `OUTLIER_CANDIDATES`, `CROSS_SPLIT_DUPLICATE`, `GROUP_OVERLAP`,
`POSSIBLE_TARGET_COPY`, `GENERALIZATION_GAP`, `WIDE_METRIC_INTERVAL`, `SYSTEMATIC_BIAS`.

Additional v0.1 codes: `INVALID_NUMERIC_VALUES`, `NEAR_CONSTANT_FEATURE`, `DUPLICATE_ROWS`,
`DISCORDANT_DUPLICATES`, `MIXED_TYPES`, `DTYPE_MISMATCH`, `MISSING_TARGET`,
`ONE_CLASS_TARGET`, `CLASS_MISSING_IN_SPLIT`, `DEGENERATE_SCALE`,
`SUSPICIOUS_TARGET_ASSOCIATION`, `SUSPICIOUS_FEATURE_NAME`, `UNDEFINED_METRIC`,
`INSUFFICIENT_VALID_REPLICATES`, `SMALL_EVALUATION_SET`, `CLASS_NEVER_PREDICTED`,
`PREDICTION_COLLAPSE`, `UNEVEN_CLASS_PERFORMANCE`, `TEST_OUTPERFORMS_TRAIN`.

Codes are stable identifiers. Titles, messages and recommendations may be reworded between
versions; codes and evidence keys change only with a schema version bump.

## Status policy

| Findings present | Status |
|------------------|--------|
| any `high` or `critical` | `failed` |
| otherwise any `low` or `medium` | `warning` |
| none, or `info` only | `passed` |

`skipped` and `error` come from applicability checks and exceptions, never from findings.
