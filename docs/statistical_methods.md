# Statistical methods

This document specifies what every diagnostic computes, the assumptions behind it, how
evidence is turned into findings, and what the method cannot tell you. Implementation
references are given as `module:function`.

**General principles**

- **Evidence comes from computation.** Rules in `recommendations/rules.py` grade computed
  numbers. They never create numbers.
- **Undefined is not zero.** Undefined quantities (constant columns, empty classes,
  `SS_tot = 0`) are `null` with a machine-readable reason.
- **Thresholds are screening heuristics.** None of them is a hypothesis test, and no finding
  is proof of a defect. All are configurable (see `core/config.py`).
- **No false passes.** A check that lacks the data it needs is `skipped` with a reason, and
  unrun sub-checks are listed in `skipped_subchecks`.
- **Determinism.** Every random draw comes from `numpy.random.default_rng([seed,
  crc32(stream)])`. Each diagnostic has its own stream, so results do not depend on which
  other checks ran.
- **Severity → status.** Any `high` or `critical` finding gives `failed`; otherwise any
  `low` or `medium` finding gives `warning`; otherwise `passed`. `info` findings never change
  the status.

Floats in reports are rounded to 10 significant digits, which keeps precision at every scale.

---

## 1. Missingness and data integrity (`data_integrity`)

Computed separately for every available split, over feature columns only.

| Quantity | Definition |
|----------|------------|
| missing fraction | `Σ_i is_missing(x_ij) / n` (ordinary NA; ±Inf is *not* NA) |
| ±Inf counts | per numeric feature, positive and negative separately |
| constant feature | exactly one distinct non-missing value |
| near-constant | most frequent value covers ≥ 99% of non-missing rows (info) |
| duplicate feature rows | `n − #distinct(feature rows)`; NA equals NA |
| duplicate full rows | as above over features + target |
| discordant duplicates | sets of identical feature rows with > 1 distinct target value |
| mixed types | text column with both numeric-parsable and non-numeric values |
| dtype mismatch | the same column parses as different kinds (numeric/boolean/text) across splits |
| missing target | count of rows with missing target |

**Rules.** missing fraction ≥ 0.05 → `HIGH_MISSINGNESS` medium; ≥ 0.30 → high. ±Inf →
`INVALID_NUMERIC_VALUES` medium. Constant → `CONSTANT_FEATURE` low. Duplicates →
`DUPLICATE_ROWS` low. Discordant duplicates → `DISCORDANT_DUPLICATES` medium: these are
*suspicious* (label noise, missing features), not invalid by default. Mixed types: medium
if most values are numeric (probable placeholder contamination), else low. Type mismatch
across splits → medium. Missing target → `MISSING_TARGET` medium.

**Assumptions and limitations.** The checks are descriptive. They neither test nor assume
any missingness mechanism (MCAR/MAR/MNAR). Duplicate detection uses exact matching after CSV
parsing. For continuous targets any difference between duplicates counts as discordant,
including measurement noise. Raw values are never reported.

## 2. Class imbalance (`class_imbalance`, classification only)

For each labelled split, over the *declared* labels (union of observed labels in train and
test), with missing targets excluded:

- `n_k`, `p_k = n_k / n`
- normalized Shannon entropy `H = −Σ_k p_k ln p_k / ln K`, `K` = number of declared
  classes, with `0 · ln 0 := 0`. `H` is undefined for `K = 1` or `n = 0`. Because `K` counts
  declared classes, a class missing from a split lowers `H`.
- majority/minority ratio: undefined (`null`) when a declared class has zero examples.

**Rule.** Among the classes present: `CLASS_IMBALANCE` when the minority count is below 20
**or** the minority proportion is below 0.05. Severity is `high` if the count is below 10 or
the proportion below 0.02, else `medium`. Example: 990:10 → count 10, proportion 0.01 →
high. One observed class → `ONE_CLASS_TARGET` (high). Declared classes absent from a split →
`CLASS_MISSING_IN_SPLIT` (medium).

**Interpretation.** Imbalance is a property of the data, not a defect. The recommendation
is about metric choice (balanced accuracy, per-class recall, macro-F1) and stratification.
Resampling is never prescribed.

## 3. Feature correlation (`feature_correlation`)

Numeric, non-boolean features of the **training** split. Target, ID, group, time and
excluded columns are removed first.

- **Pearson r** on pairwise-complete rows (both values finite), computed from masked
  sufficient statistics of mean-centred data so that large offsets do not cause
  cancellation (`statistics/correlations.py:_masked_pearson`). Verified against
  `scipy.stats.pearsonr`.
- **Spearman ρ** = Pearson r of average ranks. For pairs without missing values ranks are
  computed once per column. For pairs with missing values ranks are recomputed on the
  pair's complete rows (exact). Beyond 20,000 such pairs the implementation falls back to
  per-column ranks and marks `spearman_exact = false`.
- **Undefined**: `n < 10` (`insufficient_pairs`) or a column constant on the pair's joint
  rows (`constant_on_pair`). Such pairs are never reported as `r = 0`.

**Rule.** `max(|r|, |ρ|) ≥ 0.90` → `HIGH_FEATURE_CORRELATION` low; `≥ 0.99` → medium. The
top 25 pairs are reported, ranked by strength. For more than 1,000 numeric features only the
first 1,000 are analysed, and this is reported.

**Interpretation.** The pairs are redundancy candidates. Correlation is not evidence of
leakage and not grounds for automatic removal. Dependence among three or more features
(e.g. `a = b + c`) is not detected.

## 4. Outlier screening (`outliers`)

Per numeric feature, finite values only. Reference statistics come from **train** and are
applied unchanged to test (`statistics/robust.py:screen_outliers`).

1. `MAD > 0`: modified z `z = 0.6745 (x − median) / MAD` (Iglewicz and Hoaglin, 1993). A
   value is a candidate when `|z| > 3.5`, and extreme when `|z| > 10`.
2. `MAD = 0`, `IQR > 0`: Tukey fences `[Q1 − 1.5 IQR, Q3 + 1.5 IQR]`. Extreme means more
   than 3 IQRs beyond the quartiles (Tukey's "far out"). Quantiles use linear interpolation
   (Hyndman–Fan type 7).
3. `MAD = IQR = 0`: **degenerate scale**. The values that differ from the median are counted
   and reported as `DEGENERATE_SCALE` (info), with no outlier claim and no division by zero.
4. Fewer than 5 finite training values: listed in `skipped_subchecks`.

**Rule (per split).** `OUTLIER_CANDIDATES` is medium if any extreme value exists, low if some
feature has ≥ 1% candidates, otherwise info.

**Properties (tested).** MAD is scale-equivariant (`MAD(cx) = |c|·MAD(x)`) and
translation-invariant. Screening is invariant to row order.

**Limitations.** Univariate only. Heavy-tailed or multimodal variables naturally produce
candidates. Candidates are for review, never automatic deletion.

## 5. Leakage indicators (`leakage`)

1. **Cross-split exact overlap.** Each feature row is serialized canonically: numbers as
   `repr(float)` with `−0.0 → 0.0`, text verbatim, booleans tagged, and missing values as an
   explicit NA token, so NA matches NA but not the string `"NaN"`. Counts and fractions are
   reported for both sides, along with how many overlapping test rows also share the
   target. Severity: high if ≥ 10% of test rows overlap, otherwise medium. Severity drops to
   low when the feature space is small (distinct training rows < 50% of training rows),
   because chance collisions are then expected.
2. **Group overlap.** `|groups_train ∩ groups_test|`, the affected test rows and fractions.
   Group identifiers are never reported. Severity depends on the declared
   `evaluation_target`:

   | evaluation_target | severity |
   |-------------------|----------|
   | `unseen_groups` | critical |
   | `unspecified` (default) | medium, with a prompt to declare the goal |
   | `unseen_rows` | info (rows are still not independent; see uncertainty) |

3. **Target copy** (training data; `POSSIBLE_TARGET_COPY`, high):
   - exact equality with the target on all jointly observed rows (≥ 10 rows);
   - regression: exact linear transform, `|r| ≥ 1 − 1e−12`;
   - classification: one-to-one recoding (feature values map bijectively to classes, e.g.
     `"yes"/"no"` for `1/0`).
4. **Near-perfect association screen** (`SUSPICIOUS_TARGET_ASSOCIATION`, medium only):
   - regression, numeric feature: `max(|r|, |ρ|) ≥ 0.98`;
   - regression, categorical feature (2 ≤ distinct ≤ n/5): correlation ratio
     `η = sqrt(SS_between / SS_total) ≥ 0.98`;
   - binary classification, continuous feature: Mann–Whitney AUC `max(AUC, 1 − AUC) ≥ 0.99`;
   - classification, categorical feature (any text/boolean feature, or a numeric feature
     with ≤ 20 distinct values; in both cases at most n/5 distinct values): class purity
     `Σ_v max_k n_vk / n ≥ 0.99` while the majority-class baseline is < 0.99. The cardinality
     cap prevents ID-like columns from being trivially "pure".
   - Continuous features with multiclass targets are *not* screened in v0.1. This is
     reported in `skipped_subchecks`.
5. **Suspicious names** (low): feature names containing the target name (≥ 3 chars) or
   matching configurable patterns (`target`, `label`, `outcome`, `leak`, `future`, `y_true`,
   `ground_truth`, `diagnos`, `result`).

**What this does not do.** High correlation with the target is never, by itself, called
leakage. Pipeline/preprocessing leakage (scalers, imputers or feature selection fit on all
data, target encoding without folds, temporal look-ahead) cannot be detected from CSV files.
The report always states this. Passing these checks does not rule out leakage.

## 6. Generalization gap (`overfitting`)

Requires labels plus training **and** held-out predictions, both joined by row ID. The same
metric is computed on both splits (`statistics/metrics.py`). A gap exists only if the metric
is defined on both; otherwise it is `null` with the reason (`UNDEFINED_METRIC`, info).

| Metric | Direction | Gap | Medium / high threshold |
|--------|-----------|-----|-------------------------|
| accuracy, balanced accuracy, macro-F1 | higher better | train − test | 0.05 / 0.15 (absolute) |
| R² | higher better | train − test | 0.10 / 0.25 (absolute) |
| MAE, RMSE | lower better | test − train | 25% / 50% of the training loss (relative) |

A positive gap means worse on held-out data. The relative loss gap is undefined when the
training loss is 0. Per-class recall gaps are reported as evidence without separate
findings, to avoid multiplicity. Held-out performance better than training by at least
the medium threshold gives `TEST_OUTPERFORMS_TRAIN` (info).

**Interpretation.** The finding is called a *generalization gap*. Overfitting is one
explanation. Others are distribution shift between splits, split errors and leakage into
training. Gaps are point estimates (see §7 for held-out uncertainty). Without training
predictions the check is skipped; it never reports "no overfitting".

**Example (tested).** Train R² = 0.95 and test R² = 0.20 give a gap of 0.75, which is high.

## 7. Performance uncertainty (`uncertainty`)

- **Row bootstrap**: draw n rows with replacement.
- **Cluster bootstrap** (when `--group` is set): draw G groups with replacement and include
  every member row of each draw, preserving multiplicity. Rows from the same participant are
  dependent, so resampling rows would understate uncertainty. In the tests, with a strong
  within-group error correlation, the cluster interval is more than 2× wider than the row
  interval, as expected from the design effect `1 + (m − 1)·ICC`.
- `B = 1000` replicates by default, from the `uncertainty` RNG stream, with the same
  replicate index sets reused for every metric. The interval is the percentile interval
  `[q_{α/2}, q_{1−α/2}]` (linear interpolation) of the valid replicate values. The point
  estimate is the full-sample metric.
- Metrics: MAE, RMSE, R² (regression); accuracy, balanced accuracy, macro-F1
  (classification).
- **Undefined replicates.** R² is undefined when the resampled target is constant. Balanced
  accuracy and macro-F1 are undefined in a replicate that lacks a class present in the full
  evaluation set, because the metric would then average over a different set of classes.
  Such replicates are dropped and counted. If fewer than 80% are valid,
  `INSUFFICIENT_VALID_REPLICATES` (medium) is raised. Because dropping replicates
  conditions the interval, a low valid fraction means the interval should not be trusted.

**Rules.** `WIDE_METRIC_INTERVAL` fires when the width exceeds 0.10 for bounded scores
(accuracy, balanced accuracy, macro-F1, R²) or 50% of the estimate for losses. It is low,
or medium when the width is more than twice the threshold. `SMALL_EVALUATION_SET` (low)
fires with fewer than 30 resampling units.

**Estimand and limitations.** The interval describes the performance of the *fixed, trained*
model on the population the evaluation set represents. It says nothing about re-training
variability, the choice of split, or future temporal drift. Percentile intervals can
under-cover with small or imbalanced evaluation sets. BCa or studentized intervals are
future work.

## 8. Prediction errors (`prediction_errors`)

**Regression.** With `e = ŷ − y`: MAE `= mean|e|`, MSE, RMSE `= sqrt(mean e²)`, median `|e|`,
mean signed error `= mean(e)`, R² `= 1 − SS_res/SS_tot` (undefined for n < 2 or a constant
target; never substituted with 0 or 1), signed-error quantiles (5/25/50/75/95%), MAE and mean
signed error by target quintile (n ≥ 25), and the least-squares slope of `y` on `ŷ`.

- `SYSTEMATIC_BIAS` (medium) when `bias_fraction = mean(e)² / mean(e²) ≥ 0.25`. Because
  `mean(e²) = mean(e)² + var(e)`, this is the share of squared error explained by a constant
  offset. It is scale-free.
- `PREDICTION_COLLAPSE` (high) when all predictions are identical while the target varies.

**Classification** (hard labels). Confusion matrix (rows = truth, columns = prediction),
support, and the explicit zero-division policy:

| quantity | definition | undefined when |
|----------|-----------|----------------|
| precision_k | TP/(TP+FP) | class never predicted |
| recall_k | TP/(TP+FN) | class absent from y |
| F1_k | 2TP/(2TP+FP+FN) | class absent from both y and ŷ |
| accuracy | mean(y = ŷ) | n = 0 |
| balanced accuracy | mean recall over classes present in y | fewer than 2 classes present |
| macro-F1 | mean F1 over classes in y ∪ ŷ (scikit-learn's label set) | no classes |

These match scikit-learn wherever scikit-learn's value is defined (property-tested). Where
scikit-learn substitutes 0 for an undefined value, this tool reports `null` with a reason.

Findings: `PREDICTION_COLLAPSE` (high; one predicted class while ≥ 2 are present),
`CLASS_NEVER_PREDICTED` (medium), `SYSTEMATIC_BIAS` (medium; a predicted class share differs
from its true share by ≥ 0.15), `UNEVEN_CLASS_PERFORMANCE` (low; a class with support ≥ 10
has recall ≤ 50% of the best class recall), `UNDEFINED_METRIC` (info) and `ONE_CLASS_TARGET`
(medium) when the evaluation set has a single class.

Patterns are described, not explained causally. No probability or calibration claims are
made.

---

## Limitations and future work

| Area | Current limitation | Possible future work |
|------|--------------------|----------------------|
| Leakage | Indicators only; no pipeline introspection | Accept pipeline metadata, fold-aware target-encoding checks, temporal look-ahead checks with `time_column` |
| Association screen | No continuous × multiclass screen | One-vs-rest AUC or mutual-information screen |
| Correlation | Pairwise only | VIF / condition-number screen for multicollinearity |
| Outliers | Univariate | Robust multivariate distance (MCD) |
| Bootstrap | Percentile only | BCa / studentized intervals; stratified bootstrap for rare classes |
| Overfitting | Point gaps | Interval for the gap (paired bootstrap across splits is not valid; needs CV) |
| Predictions | Hard labels only | Probability inputs: calibration, Brier score, ROC/PR with intervals |
| Inputs | CSV only | Parquet, chunked reading for very large files |
| Shift | Not covered | Train/test covariate shift tests (e.g. classifier two-sample test) |
| Multiplicity | Many screens at fixed thresholds | Optional false-discovery control for screening-style findings |
