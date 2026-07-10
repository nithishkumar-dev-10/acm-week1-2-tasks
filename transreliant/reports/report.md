# TransReliant Cascade — Project Report

## 1. Problem Description

This project predicts train ticket outcomes for the Indian Railway booking
system using a two-stage cascade, rather than a single model:

- **Model 1 (Classification):** predicts `Confirmation Status` (Confirmed /
  Not Confirmed) from booking and journey features.
- **Model 2 (Regression):** for passengers Model 1 flags as **Not
  Confirmed**, predicts `Waitlist Position` (1–200) — an estimate of how
  bad the waitlist situation is.

Model 2 only ever runs on the subset Model 1 routes to it. This mirrors the
brief's own Hospital Triage example: Model 1 classifies risk, Model 2
estimates severity only for the flagged subset. We don't waste a regression
model estimating waitlist severity for passengers who are already
confirmed — we only run it for the people who actually need to know how
bad their wait is.

## 2. Dataset

- **Source:** Indian Railway Ticket Confirmation dataset —
  https://www.kaggle.com/datasets/aaryananil/indian-railway-ticket-confirmation
- **Size (post feature engineering):** 30,000 rows, 22 columns
  (`data/processed/featured.csv`)
- **Targets:**
  - Stage 1: `Confirmation Status` (0 = Not Confirmed, 1 = Confirmed)
  - Stage 2: `Waitlist Position` (continuous, 1–200), trained only on the
    ground-truth Not-Confirmed subset (10,053 of 30,000 rows)
- **Dropped columns:** `Train Number`, `PNR Number`, `Booking Channel`,
  `Current Status` — `Train Number` is a near-unique ID with no
  generalizable signal; the others were excluded per the project's own
  feature-selection decisions.

**Why a second, originally-considered delay dataset was rejected:** an
earlier version of this project considered joining a separate delay
dataset for a feature-passing cascade. That dataset shared no common key
with the ticket confirmation data and was far too small (100 rows) to
train a reliable regressor. The final design instead uses one real,
well-sized, clean dataset for both stages, avoiding any artificial
linking.

## 3. EDA Summary

Findings below are drawn from `notebooks/eda.ipynb`:

- **Class balance:** `Confirmation Status` is imbalanced roughly 2:1 —
  66.5% Confirmed, 33.5% Not Confirmed. This ruled out plain accuracy as
  a Stage 1 metric in favor of ROC-AUC / F1 on the minority class.
- **`Waitlist Position` distribution:** null for ~67% of rows (Confirmed
  passengers legitimately have no waitlist number). On the genuinely
  Not-Confirmed subset (10,053 rows), it's close to uniform across
  1–200 (mean ≈ 99.4, median = 100, std ≈ 58.0) rather than concentrated
  near 1 the way a real queue would be — this foreshadows Stage 2's
  near-zero R² (Section 7).
- **`days_before_journey` is a constant.** `Date of Journey` equals
  `Booking Date` + 244 days for all 30,000 rows with zero exceptions.
  This makes `days_before_journey` zero-variance (correlation with the
  target undefined) and collapses `booking_urgency_bucket` into a single
  "early" category for every row — both are dead features in this
  dataset, not just weak ones.
- **Correlation check:** every remaining numeric feature (`Seat
  Availability`, `Travel Distance`, `Number of Stations`, `Travel Time`,
  `Number of Passengers`, `journey_month`, `journey_dayofweek`) has
  |r| < 0.01 with `Confirmation Status`. This is the concrete root cause
  behind Stage 1's chance-level AUC (Section 6), not a modeling gap.
- **`Current Status` is a leakage column** — it maps perfectly onto
  `Confirmation Status` (Waitlisted → always Not Confirmed; RAC/Confirmed
  → always Confirmed), which is why it's dropped rather than engineered.
- **`Train Number`** (25,553 unique / 30,000 rows) and **`PNR Number`**
  (unique per row) are ID-like with no generalizable signal — dropped.
- **`Age of Passengers`**, **`Holiday or Peak Season`**, and **`Booking
  Channel`** are all roughly even categorical splits with no standout
  skew.

## 4. Data Cleaning & Feature Engineering

**`seat_pressure`** — computed as `Number of Passengers / (Seat
Availability + 1)`. The `+1` is purely a div-by-zero guard, not a
correction for messy data — `Seat Availability` ranges 0–499 with no
negative values. The feature is right-skewed as expected (most bookings
have low pressure, a long tail of high-pressure ones), but group means
between Confirmed and Not-Confirmed are nearly identical (0.078 vs 0.079),
so separation is weak. It's kept anyway as the most logically-grounded
engineered feature for Stage 1, without expecting it to carry the model
on its own.

**`booking_urgency_bucket`** — binned from `days_before_journey` into
`last_minute` / `short` / `planned` / `early`. In this dataset it's
effectively a dead feature: `Date of Journey` is always exactly
`Booking Date` + 244 days for all 30,000 rows (zero variance, confirmed
in EDA), so every row falls into the same `early` bucket after binning.
It one-hot-encodes into a single always-1 column that contributes nothing
to either model. The binning logic itself isn't broken — there's simply
no booking-lead-time variation in this dataset to bucket. Kept in the
pipeline for structural completeness and to match what the assignment
expects, with this limitation documented rather than silently dropped.

**Why `Waitlist Position` is excluded from the Stage 2 feature set** — it
is the Stage 2 *target*, not an input feature. Including it as a feature
would be direct label leakage: the model would trivially learn to copy
its own prediction target rather than learning any real relationship
between booking/journey characteristics and waitlist severity.

Two other engineered features are also in the set: `route_length_per_stop`
(`Travel Distance / (Number of Stations + 1)`) showed no separation with
`Confirmation Status` either, but was kept mainly for Stage 2 on the
chance route geography correlates with waitlist severity — untested.
`is_peak_or_holiday` (binary flag from `Holiday or Peak Season`) showed
negligible separation (~66.7% Confirmed regardless of the flag) but was
cheap to keep.

## 5. Pipeline Architecture

![Pipeline Architecture](figures/pipeline_arch.png)

The cascade is **routing-based**, not feature-passing: Model 1's output is
a routing decision (does this row go to Model 2 at all?), not a feature
fed into Model 2's input space. This is a legitimate and arguably cleaner
cascade variant than passing Model 1's prediction as a Model 2 feature,
since it avoids compounding Model 1's errors directly into Model 2's
inputs — Model 2 only ever sees the original booking/journey features,
never Model 1's (often noisy) prediction.

## 6. Stage 1 Evaluation — Confirmation Status Classifier

**Data split:** 24,000 train / 6,000 test rows (stratified), train class
balance 66.5% Confirmed.

**Baseline model comparison (5-fold CV):**

| Model | F1 | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.5904 (±0.0082) | 0.5054 (±0.0052) |
| Random Forest | 0.7489 (±0.0035) | 0.5041 (±0.0046) |
| XGBoost | 0.6121 (±0.0104) | 0.5072 (±0.0091) |

Note Random Forest's high F1 despite AUC near chance level — this is the
class-imbalance trap: with 66.5% positive class, leaning toward
"Confirmed" buys F1 without any real discrimination. Model selection was
therefore done on **ROC-AUC**, not F1, since AUC can't be gamed by class
imbalance the same way.

**Tuned models (RandomizedSearchCV, selected on CV AUC):**

| Model | CV AUC | CV F1 |
|---|---|---|
| Logistic Regression (tuned) | 0.5054 (±0.0054) | 0.5898 (±0.0087) |
| Random Forest (tuned) | 0.5070 (±0.0065) | 0.5875 (±0.0155) |
| XGBoost (tuned) | 0.5070 (±0.0039) | 0.6398 (±0.0056) |

**Selected model: XGBoost** (best tuned CV AUC = 0.5070, narrowly ahead of
Random Forest at 0.5070 — tie broken by XGBoost's higher CV F1).

**Held-out test performance:**
- Test F1 (default 0.5 threshold): 0.6354
- Test AUC: 0.4992
- Confusion matrix at the selected operating threshold (0.6): `[[1536, 475], [3002, 987]]`

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Not Confirmed (0) | 0.34 | 0.76 | 0.47 | 2011 |
| Confirmed (1) | 0.68 | 0.25 | 0.36 | 3989 |
| **Accuracy** | | | **0.4205** | 6000 |

**Threshold selection (Step 12):** the same precision-floor selection
logic from Step 12 was re-run against the retrained models. The fallback
landed on **threshold = 0.6** this run (precision = 0.3385, recall =
0.7638) rather than the 0.55 seen in an earlier training run — retraining
shifted the precision/recall trade-off slightly, so the non-degenerate
operating point moved with it. `config.yaml` reflects the current value
(0.6).

**Finding:** Test AUC (0.4992) is at/near random chance (0.50). This
matches a raw-data audit showing no available feature is statistically
related to `Confirmation Status`. This is treated as a **dataset finding**,
not a modeling failure — see Limitations (section 10).

![Confusion Matrix](figures/confusion_matrix_stage1.png)
![ROC Curve](figures/roc_stage1.png)
![PR Curve](figures/pr_curve_stage1.png)
![Feature Importance](figures/feature_importance_stage1.png)

## 7. Stage 2 Evaluation — Waitlist Position Regressor

**Data:** 10,053 genuinely Not-Confirmed rows (of 30,000 total). Split:
8,042 train / 2,011 test. Target range 1–200, mean ≈ 99.4.

**Baseline model comparison (5-fold CV, RMSE):**

| Model | CV RMSE |
|---|---|
| Ridge | 60.0509 (±0.6108) |
| Random Forest | 58.9017 (±0.2779) |
| XGBoost | 60.7462 (±1.0067) |

**Selected model: Random Forest** (lowest baseline CV RMSE among the
three compared).

> **Note on model selection:** an earlier training run selected **Ridge**
> as the Stage 2 winner (CV RMSE ≈ 58.24 at the time). On this retrained
> run, Random Forest came out ahead at the baseline stage and was carried
> through tuning instead. This is a legitimate re-run outcome — CV scores
> for near-random targets sit close enough together that small changes in
> resampling can flip the ranking — but it means `artifacts/models/
> stage2_regressor.pkl` is now a tuned `RandomForestRegressor`, not
> `Ridge`. Anyone loading that artifact directly should expect that type.

**Tuned model (RandomizedSearchCV):**

| Model | CV RMSE |
|---|---|
| Random Forest (tuned) | 57.8775 (±0.4366) |

Best params: `max_depth=3, min_samples_leaf=1, n_estimators=600`.

**Held-out test performance:**
- Test RMSE: 58.5685
- Test MAE: 51.1669
- Test R²: -0.0010

**Finding:** Test R² (-0.0010) is at/near zero — the model explains almost
none of the variance in `Waitlist Position`. This lines up with the EDA
observation that `Waitlist Position` looks close to uniformly distributed
across 1–200 rather than driven by booking/journey features the way a real
waitlist queue would be. Treated as a dataset finding, not a modeling
failure (see Limitations).

![Predicted vs Actual](figures/pred_vs_actual_stage2.png)
![Residuals](figures/residuals_stage2.png)

## 8. System-Level Evaluation

Running the full cascade end-to-end over the Stage 1 test set (Step 20)
gives:

- **System ROC-AUC:** 0.4992 — matches Stage 1's own held-out AUC
  (Section 6), as expected, since the end-to-end classification decision
  is still Stage 1's.
- **Coverage (% of test set routed to Stage 2):** 75.63% (n=1,536 of
  6,000) — well above the dataset's true Not-Confirmed rate of 33.5%.
  This gap is a direct consequence of the threshold=0.6 operating point:
  at that threshold, precision on the Not-Confirmed class is only 0.3385
  (Section 6), so Stage 1 is routing a large number of actually-Confirmed
  passengers into Stage 2 as false positives, inflating coverage well past
  the true Not-Confirmed proportion. This gap is larger than the 51.42%
  coverage seen on the earlier 0.55-threshold run — a direct effect of the
  new threshold's much higher Not-Confirmed recall (0.7638 vs. the earlier
  run's 0.5301) pulling more passengers into Stage 2.
- **Stage 2 RMSE on the Stage-1-routed subset:** 57.4637 (n=1,536), vs.
  58.5685 on Stage 2's own ground-truth-clean test split (Section 7). The
  routed-subset number is slightly *lower*, which is a bit counter-
  intuitive given it includes Stage 1's misclassifications — but with both
  numbers this close to the ~58 RMSE ceiling of a near-random model
  (Section 7's R² ≈ -0.0010), this difference isn't meaningful signal, just
  noise around a model that isn't predicting much of anything either way.


## 9. Justification for Multi-Model Design

A single model cannot do both jobs here because Stage 1 and Stage 2 solve
fundamentally different problem types:

- **Different target types.** `Confirmation Status` is binary
  (classification); `Waitlist Position` is a bounded continuous value
  (regression). No single scikit-learn/XGBoost estimator natively
  optimizes both a classification loss and a regression loss on the same
  output head without a custom multi-task architecture — unnecessary
  complexity for a problem this size.
- **Conditional relevance.** `Waitlist Position` is only a meaningful
  quantity for passengers who are *not* confirmed — it's undefined
  (or at least uninformative) for confirmed passengers. A single combined
  model would waste capacity and training signal trying to also predict a
  waitlist severity for rows where that number doesn't apply.
- **Routing, not feature-passing.** Because Model 2 is conditionally
  relevant only to Model 1's "Not Confirmed" output, a *routing-based*
  cascade (Section 5) is the correct structure: Model 1's job is partly to
  decide "does this row even need a Model 2 prediction," not just to
  produce a feature for Model 2 to consume.

