#!/usr/bin/env bash
# Regenerate the example datasets and their sample reports (run from the repository root).
set -euo pipefail
python examples/generate_examples.py

cd examples/regression
mldoctor diagnose --train train.csv --test test.csv \
  --predictions test_predictions.csv --train-predictions train_predictions.csv \
  --target price --row-id row_id --task regression \
  --output sample_report.json --quiet || true
cd ../classification
mldoctor diagnose --train train.csv --test test.csv \
  --predictions test_predictions.csv --train-predictions train_predictions.csv \
  --target churned --row-id customer_id --task classification \
  --output sample_report.json --quiet || true
cd ../grouped_data
mldoctor diagnose --train train.csv --test test.csv \
  --predictions test_predictions.csv --train-predictions train_predictions.csv \
  --target gait_speed --row-id visit_id --group participant_id \
  --evaluation-target unseen_groups --task regression \
  --output sample_report.json --quiet || true
