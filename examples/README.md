# Example datasets

All examples are synthetic and deterministic: `python examples/generate_examples.py`
regenerates them (seed 2026), and `examples/run_examples.sh` regenerates them together with
each `sample_report.json`. Predictions are **simulated** with a known error structure. No
model is trained, so every planted issue has a known ground truth.

| Folder | Task | Planted issues | Expected findings |
|--------|------|----------------|-------------------|
| `regression/` | house price | `sqft_m2 = 0.0929 × sqft`; 35% missing `garage_area` in test; one `lot_size` entry error; 12 train rows copied into test (some later masked by missingness); near-perfect train predictions; +15k bias on test predictions | `HIGH_MISSINGNESS` (high, test), `HIGH_FEATURE_CORRELATION`, `OUTLIER_CANDIDATES` (extreme), `CROSS_SPLIT_DUPLICATE`, `GENERALIZATION_GAP` (high), `SYSTEMATIC_BIAS` |
| `classification/` | churn (binary) | ~6% positives; `refund_issued` is a yes/no recoding of the label; constant `region_code`; 8% missing `notes_len`; memorised training labels; weak recall on churners | `POSSIBLE_TARGET_COPY`, `CLASS_IMBALANCE`, `CONSTANT_FEATURE`, `HIGH_MISSINGNESS`, `GENERALIZATION_GAP`, `UNEVEN_CLASS_PERFORMANCE`, `WIDE_METRIC_INTERVAL` |
| `grouped_data/` | gait speed, 8 visits per participant | 6 participants in both splits; per-participant error offsets | `GROUP_OVERLAP` (critical with `--evaluation-target unseen_groups`), `CROSS_SPLIT_DUPLICATE`, cluster bootstrap over 21 participants, `WIDE_METRIC_INTERVAL` |

## Commands

```bash
cd examples/regression
mldoctor diagnose --train train.csv --test test.csv \
  --predictions test_predictions.csv --train-predictions train_predictions.csv \
  --target price --row-id row_id --task regression

cd ../classification
mldoctor diagnose --train train.csv --test test.csv \
  --predictions test_predictions.csv --train-predictions train_predictions.csv \
  --target churned --row-id customer_id --task classification

cd ../grouped_data
mldoctor diagnose --train train.csv --test test.csv \
  --predictions test_predictions.csv --train-predictions train_predictions.csv \
  --target gait_speed --row-id visit_id --group participant_id \
  --evaluation-target unseen_groups --task regression

# Same data, but the goal is generalization to new visits of known participants:
mldoctor leakage --train train.csv --test test.csv --target gait_speed \
  --group participant_id --evaluation-target unseen_rows --task regression
```

The prediction files are deliberately shuffled relative to the data files. Joining is by
row ID, so the order does not matter.

`config.example.toml` lists every configuration option with its default.
