<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,50:203a43,100:2c5364&height=200&section=header&text=TransReliant%20Cascade&fontSize=60&fontColor=ffffff&fontAlignY=38&desc=Predictive%20Reliability%20for%20Ticket%20Confirmation%20and%20Waitlist%20Severity&descAlignY=58&descSize=18&descColor=90cdf4&animation=fadeIn" width="100%"/>

<br/>

```
  🚦  Classify · Route · Regress  🎯
```

<img src="https://img.shields.io/badge/Task-Classification%20%2B%20Regression-0ea5e9?style=flat-square&labelColor=0f172a" />
&nbsp;
<img src="https://img.shields.io/badge/Architecture-Two--Stage%20Cascade-8b5cf6?style=flat-square&labelColor=0f172a" />
&nbsp;
<img src="https://img.shields.io/badge/Models-Compared%20%26%20Tuned-10b981?style=flat-square&labelColor=0f172a" />
&nbsp;
<img src="https://img.shields.io/badge/Status-Completed-22c55e?style=flat-square&labelColor=0f172a" />

<br/><br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Classifier%20%2B%20Regressor-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)

<br/>

[![Dataset](https://img.shields.io/badge/📊%20Dataset-Indian%20Railway%20Ticket%20Confirmation-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/aaryananil/indian-railway-ticket-confirmation)
[![License](https://img.shields.io/badge/License-MIT-a855f7?style=for-the-badge)](LICENSE)

<br/>

---

<table border="0" width="85%">
<tr>
<td align="center" width="25%" style="padding: 10px">
<br/>
<b>🎯 Task</b><br/>
Classification → Regression<br/>
<code>two-stage cascade</code>
</td>
<td align="center" width="25%" style="padding: 10px">
<br/>
<b>📦 Dataset</b><br/>
Indian Railway Tickets<br/>
<code>30,000 rows · 22 cols</code>
</td>
<td align="center" width="25%" style="padding: 10px">
<br/>
<b>🔬 Stage 1</b><br/>
LR · RF · XGBoost<br/>
<code>Confirmation Status</code>
</td>
<td align="center" width="25%" style="padding: 10px">
<br/>
<b>🔬 Stage 2</b><br/>
Ridge · RF · XGBoost<br/>
<code>Waitlist Position</code>
</td>
</tr>
</table>

<br/>

> *"Users lack a data-driven reliability metric before planning journeys."*
>
> **TransReliant Cascade** answers that directly: a classifier predicts whether a
> booking will confirm, and — for the ones that won't — a regressor estimates
> exactly how severe the waitlist situation will be.

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f2027,100:2c5364&height=3&section=header" width="85%"/>

<br/>

</div>

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Approach](#-approach)
- [Project Structure](#-project-structure)
- [ML Pipeline Overview](#️-ml-pipeline-overview)
- [Feature Engineering](#-feature-engineering)
- [Cascade Design](#-cascade-design)
- [Stage-Wise Evaluation](#-stage-wise-evaluation)
- [Results](#-results)
- [Threshold Optimization](#-threshold-optimization)
- [Quickstart](#-quickstart)
- [Configuration](#️-configuration)
- [Artifacts](#-artifacts)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)

---

## 🎯 Problem Statement

Indian Railways passengers often face uncertainty in ticket confirmations and waitlist outcomes before planning their journeys. TransReliant addresses this challenge by providing data-driven predictions for ticket confirmation reliability and estimated waitlist severity, enabling passengers to make more informed travel decisions.

---

## 🧩 Approach

TransReliant answers this with a two-stage system built on the Indian Railway ticket confirmation data:

- **Stage 1** predicts whether a booking will be **Confirmed** or **Not Confirmed**, using booking and journey features.
- **Stage 2** runs only for passengers Stage 1 flags as **Not Confirmed**, and predicts their **Waitlist Position** — turning a binary outcome into a concrete reliability estimate.

This gives users a single, data-driven answer to the question the problem statement raises: not just *will my ticket confirm*, but *if not, how bad will it be*.

<div align="center">
<img src="reports/figures/architecture.png" alt="TransReliant Cascade architecture diagram" width="90%"/>
</div>

---

## 📁 Project Structure

```
TransReliant/
│
├── data/
│   ├── raw/
│   │   └── ticket_confirmation.csv
│   └── processed/
│       ├── cleaned.csv
│       ├── featured.csv
│       └── splits/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_feature_engineering_exp.ipynb
│
├── src/
│   ├── config_loader.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── train_stage1.py
│   ├── train_stage2.py
│   ├── evaluate.py
│   ├── pipeline.py
│   └── utils.py
│
├── artifacts/
│   └── models/
│       ├── preprocessor_stage1.pkl
│       ├── preprocessor_stage2.pkl
│       ├── stage1_classifier.pkl
│       └── stage2_regressor.pkl
│
├── reports/
│   ├── figures/
│   ├── metrics/
│   └── report.md
│
├── logs/
│   └── experiment_log.csv
│
├── config.yaml
├── requirements.txt
├── main.py
└── run_pipeline.py
```

---

## ⚙️ ML Pipeline Overview

```
Raw CSV (Indian Railway Ticket Confirmation)
        │
        ▼
┌───────────────────┐
│  Cleaning &        │  → Date parsing, leakage-safe Waitlist/Confirmation
│  Leakage Guards     │    handling, drop of near-unique / circular columns
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Feature Engineering │  → seat_pressure, booking_urgency_bucket,
│                    │    route_length_per_stop, is_peak_or_holiday
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Stage 1 Split      │  → Stratified 80/20, full dataset
└────────┬──────────┘
         │
         ▼
┌───────────────────────────────────────────────┐
│   Stage 1: Classification (Confirmation Status) │
│   LR / RF / XGBoost compared → tuned → thresholded │
└────────────────────────┬──────────────────────┘
                         │
                         ▼ (Not Confirmed subset only)
┌───────────────────────────────────────────────┐
│   Stage 2: Regression (Waitlist Position)       │
│   Ridge / RF / XGBoost compared → tuned         │
└────────────────────────┬──────────────────────┘
                         │
                         ▼
  📊  Cascade Output → confirmation prediction + confidence
                      + estimated waitlist position (if flagged)
```

---

## 🔧 Feature Engineering

| Feature | Logic | Why It Helps |
|---|---|---|
| `seat_pressure` | `Number of Passengers / (Seat Availability + 1)` | Direct proxy for booking pressure — the strongest engineered signal in the pipeline |
| `booking_urgency_bucket` | Binned `days_before_journey`: last-minute / short / planned / early | Captures the non-linear relationship between how late a booking is and waitlist risk |
| `route_length_per_stop` | `Travel Distance / (Number of Stations + 1)` | Normalizes journey length against route density |
| `is_peak_or_holiday` | Binarized peak/holiday flag | Direct seasonal demand signal |
| `journey_month`, `journey_dayofweek`, `days_before_journey` | Derived from journey and booking dates | Captures seasonality and booking-lead-time effects |

All statistics used for encoding/scaling are fit on the training split only and applied unchanged to validation/test data.

---

## 🧠 Cascade Design

- **Stage 1 (Classification):** Predicts `Confirmation Status` from booking and journey features. Tuned via cross-validated comparison of Logistic Regression, Random Forest, and XGBoost, selected on ROC-AUC (not F1 — F1 is misleading here given the class imbalance), followed by hyperparameter search on the winner.
- **Stage 2 (Regression):** Predicts `Waitlist Position` for passengers **actually** not confirmed in the training data — never on Stage 1's own predictions, to avoid compounding classification errors into the regression target.
- **Inference-time routing:** Stage 2 only executes for passengers Stage 1 flags as "Not Confirmed." This is the literal cascade wiring — Stage 1's output decides *whether* Stage 2 runs, not what features it sees.
- **Independent preprocessing:** Each stage fits its own `ColumnTransformer`, since Stage 2 trains on a smaller, distributionally different subset — sharing a fitted preprocessor across stages would leak Stage 1's population statistics into Stage 2.

---

## 📊 Stage-Wise Evaluation

| Stage | Metrics Tracked | Reported In |
|---|---|---|
| Stage 1 — Classification | Precision, Recall, F1, ROC-AUC, confusion matrix, feature importance | `reports/metrics/stage1_metrics.csv` |
| Stage 2 — Regression | RMSE, MAE, R², predicted-vs-actual, residual plot | `reports/metrics/stage2_metrics.csv` |
| System — End-to-End | Cascade-level classification metrics, Stage 2 error on the actually-routed subset, coverage (% of test set routed to Stage 2) | `reports/metrics/system_metrics.csv` |

Every model comparison (Stage 1 and Stage 2) is logged run-by-run — not just the final winner — in [`logs/experiment_log.csv`](logs/experiment_log.csv).

### 📈 Results

*(Test set, n = 6,000. Source: `reports/metrics/stage1_metrics.csv`, `stage2_metrics.csv`. `system_metrics.csv` — coverage — not yet generated/uploaded.)*

| Metric | Stage 1 (Classification) | Stage 2 (Regression) |
|---|---|---|
| Primary metric | ROC-AUC: **0.508** | RMSE: **58.60** |
| Accuracy | 0.506 | — |
| Precision / Recall (Not Confirmed) | 0.346 / 0.530 | — |
| F1 (Not Confirmed) | 0.418 | — |
| MAE | — | 51.18 |
| R² | — | **-0.002** |
| Coverage (% routed to Stage 2) | `TODO` — needs `system_metrics.csv` (test set is 33.5% "Not Confirmed" by actual label, n=2,011/6,000, but coverage should be measured off *predicted* routing at threshold 0.55, not the true label) | same |

> ⚠️ **Honest finding:** Stage 1 ROC-AUC (0.508) is statistically indistinguishable
> from a coin flip, and Stage 2 R² (-0.002) means the regressor performs
> *slightly worse* than just predicting the mean waitlist position for every
> passenger. The engineered features (`seat_pressure`, `booking_urgency_bucket`,
> `route_length_per_stop`, `is_peak_or_holiday`) currently carry effectively no
> predictive signal for either target in this dataset. This is a real, useful
> result to report as-is — it validates that the cascade *architecture* (routing,
> leakage-safe splits, independent preprocessing) is correctly wired end-to-end,
> but it means the *feature set* does not yet explain confirmation status or
> waitlist severity. The "Status: Completed ✅" banner above refers to the
> pipeline being built and runnable, not to strong predictive performance —
> that is the next problem to solve, likely via richer features (e.g. per-route
> or per-train historical confirmation rates) rather than more model tuning.

---

## 🎯 Threshold Optimization

The default 0.5 decision threshold is not used. Since a missed "Not Confirmed" passenger (someone blindsided with no waitlist estimate) is costlier than a false alarm (Stage 2 runs unnecessarily), the threshold is optimized for **recall on the "Not Confirmed" class**, subject to a minimum precision constraint, rather than for raw F1. The chosen threshold — **0.55** — is stored in `config.yaml`, not hardcoded in source.

---

## 🚀 Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Add the Dataset

Download the dataset and place it at:

```
data/raw/ticket_confirmation.csv
```

> 📎 Dataset: [Indian Railway Ticket Confirmation — Kaggle](https://www.kaggle.com/datasets/aaryananil/indian-railway-ticket-confirmation)

### 3. Run the Pipeline

```bash
python run_pipeline.py
```

This regenerates cleaned/featured data, both trained pipelines, all figures, and all metrics from a clean `artifacts/` and `data/processed/` state.

### 4. Run the Interactive Demo

```bash
python main.py
```

Prompts for the highest-impact booking fields (class of travel, quota, travel distance, days before journey, seat availability, passenger count) and auto-fills the rest, then prints:

```
Confirmation Prediction : Not Confirmed (confidence 78%)
Estimated Waitlist Pos. : 63 (out of ~200)
```

---

## 🛠️ Configuration

| Setting | Value | Notes |
|---|---|---|
| Train/test split | 80% / 20% | `random_state=42`, stratified on Stage 1 target |
| Model family, Stage 1 | Best of LR / RF / XGBoost | Selected by cross-validated ROC-AUC |
| Model family, Stage 2 | Best of Ridge / RF / XGBoost | Selected by cross-validated RMSE |
| Decision threshold | **0.55**, stored in `config.yaml` | Optimized for recall on "Not Confirmed," subject to a precision floor |
| External data | None | All features derived from the single ticket confirmation dataset |

---

## 📦 Artifacts

| Artifact | Path | Description |
|---|---|---|
| Stage 1 preprocessor | `artifacts/models/preprocessor_stage1.pkl` | Fitted `ColumnTransformer` for Stage 1, joblib-serialized |
| Stage 1 classifier | `artifacts/models/stage1_classifier.pkl` | Tuned classifier, joblib-serialized |
| Stage 2 preprocessor | `artifacts/models/preprocessor_stage2.pkl` | Fitted `ColumnTransformer` for Stage 2, joblib-serialized |
| Stage 2 regressor | `artifacts/models/stage2_regressor.pkl` | Tuned regressor, joblib-serialized |

Each stage saves its preprocessor and model as separate files — Stage 2 fits its own `ColumnTransformer` rather than reusing Stage 1's, so all four artifacts are required for prediction.

**Using the saved cascade for prediction:**

```python
import joblib
from src.config_loader import load_config
from src.pipeline import predict_cascade

cfg = load_config("config.yaml")

preprocessor_stage1 = joblib.load("artifacts/models/preprocessor_stage1.pkl")
stage1_model = joblib.load("artifacts/models/stage1_classifier.pkl")
preprocessor_stage2 = joblib.load("artifacts/models/preprocessor_stage2.pkl")
stage2_model = joblib.load("artifacts/models/stage2_regressor.pkl")

results = predict_cascade(
    new_bookings_df,
    preprocessor_stage1, stage1_model,
    preprocessor_stage2, stage2_model,
    cfg,
)
# results["confirmation_prediction"], results["estimated_waitlist_position"]
```

---

## 📂 Dataset

| Property | Value |
|---|---|
| Source | [Indian Railway Ticket Confirmation — Kaggle](https://www.kaggle.com/datasets/aaryananil/indian-railway-ticket-confirmation) |
| Stage 1 Target | `Confirmation Status` (Confirmed / Not Confirmed) |
| Stage 2 Target | `Waitlist Position` (1–200), Not-Confirmed subset only |
| Records | 20,045 rows |
| Dropped Columns | `Train Number` (near-unique ID), `PNR Number`, `Booking Channel`, `Current Status` |
| Rejected Data | A separate 100-row delay dataset — no shared key with the ticket data and too small to train a reliable regressor |

---

## 🧱 Tech Stack

<div align="center">

| Layer | Tool |
|---|---|
| Language | Python 3.10+ |
| ML Frameworks | scikit-learn, XGBoost |
| Data | pandas, NumPy |
| Visualization | matplotlib, seaborn |
| Config | PyYAML |
| Serialization | joblib |

</div>

---

<div align="center">

**Predictive reliability for railway bookings — from confirmation to waitlist severity.**

</div>
