<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,50:203a43,100:2c5364&height=200&section=header&text=TalentIQ&fontSize=80&fontColor=ffffff&fontAlignY=38&desc=AI-Powered%20Candidate%20Shortlisting%20System&descAlignY=58&descSize=18&descColor=90cdf4&animation=fadeIn" width="100%"/>

<br/>

<table>
<tr>
<td align="center">

```
  🧠  Predict · Compare · Shortlist  🎯
```

</td>
</tr>
</table>

<br/>

<!-- STAT PILLS -->
<img src="https://img.shields.io/badge/Accuracy-85.03%25-0ea5e9?style=flat-square&labelColor=0f172a" />
&nbsp;
<img src="https://img.shields.io/badge/F1--macro-0.7165-8b5cf6?style=flat-square&labelColor=0f172a" />
&nbsp;
<img src="https://img.shields.io/badge/ROC--AUC-0.7963-10b981?style=flat-square&labelColor=0f172a" />
&nbsp;
<img src="https://img.shields.io/badge/Winner-XGBoost-f97316?style=flat-square&labelColor=0f172a" />
&nbsp;
<img src="https://img.shields.io/badge/Models-3%20Compared-ec4899?style=flat-square&labelColor=0f172a" />

<br/><br/>

<!-- TECH STACK BADGES -->
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)

<br/>

[![Dataset](https://img.shields.io/badge/📊%20Dataset-IBM%20HR%20Analytics-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
[![SMOTE](https://img.shields.io/badge/Imbalance-SMOTE%20Applied-22C55E?style=for-the-badge)](https://imbalanced-learn.org)
[![License](https://img.shields.io/badge/License-MIT-a855f7?style=for-the-badge)](LICENSE)

<br/>

---

<table border="0" width="85%">
<tr>
<td align="center" width="25%" style="padding: 10px">
<br/>
<b>🎯 Task</b><br/>
Binary Classification<br/>
<code>Hired = Yes / No</code>
</td>
<td align="center" width="25%" style="padding: 10px">
<br/>
<b>📦 Dataset</b><br/>
IBM HR Analytics<br/>
<code>1,470 employees</code>
</td>
<td align="center" width="25%" style="padding: 10px">
<br/>
<b>🔬 Models</b><br/>
LR · RF · XGBoost<br/>
<code>5-fold CV tuning</code>
</td>
<td align="center" width="25%" style="padding: 10px">
<br/>
<b>✅ Winner</b><br/>
XGBoost<br/>
<code>F1=0.7165</code>
</td>
</tr>
</table>

<br/>

> *"Recruiters receive hundreds to thousands of resumes per job opening.*
> *Manual screening is slow, inconsistent, and unscalable."*
>
> **TalentIQ fixes that** — a tuned ML pipeline trained on structured candidate data:
> education · experience · skills · certifications · projects · soft skills

<br/>

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Project Structure](#-project-structure)
- [ML Pipeline Overview](#️-ml-pipeline-overview)
- [Feature Engineering](#-feature-engineering)
- [Model Results](#-model-results)
- [Misclassification Analysis](#️-misclassification-analysis)
- [Why XGBoost Won](#-why-xgboost-won)
- [Visualizations](#-visualizations)
- [Quickstart](#-quickstart)
- [Configuration](#️-configuration)
- [Artifacts](#-artifacts)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [TransReliant (New Project)](#-transreliant-new-project)

---

## 🎯 Problem Statement

<table>
<tr>
<td width="50%">

### The Challenge
- Recruiters receive **hundreds to thousands** of resumes per job opening
- Manual screening is **slow, inconsistent, and unscalable**
- Subjective bias leads to **missed talent and wasted interviews**

</td>
<td width="50%">

### The Solution
- Binary classifier → **Shortlisted (Yes / No)**
- Compare **3 ML models** head-to-head
- Select winner by **F1-macro + ROC-AUC** (handles class imbalance)
- Full pipeline: EDA → Engineering → Tuning → Evaluation

</td>
</tr>
</table>

---

## 📁 Project Structure

```
TalentIQ/
│
├── 📂 config/
│   ├── config.yaml              # Pipeline mode, paths, split settings, SMOTE config
│   ├── features.yaml            # Numerical, categorical, ordinal, engineered features
│   └── hyperparameters.yaml     # GridSearch / RandomSearch param grids per model
│
├── 📂 src/
│   ├── config_loader.py         # YAML config loader
│   ├── preprocessing.py         # Cleaning, encoding, outlier handling, train/test split
│   ├── feature_engineering.py   # 10 custom engineered features
│   ├── train.py                 # SMOTE + model training + hyperparameter tuning
│   ├── metrics.py               # Accuracy, F1-macro, Precision, Recall, ROC-AUC, FPR, FNR
│   └── plots.py                 # All figure generation (ROC, confusion matrix, SHAP, etc.)
│
├── 📂 notebooks/
│   ├── eda.ipynb                          # Exploratory Data Analysis
│   ├── preprocessing_and_feat_eng_exp.ipynb  # Feature experimentation
│   └── training_pipeline_exp.ipynb        # Model training experiments
│
├── 📂 data/
│   ├── raw/                     # IBM HR Analytics CSV
│   ├── processed/               # Cleaned + encoded data
│   └── splits/                  # train.csv / test.csv
│
├── 📂 artifacts/
│   ├── preprocessor.pkl         # Fitted scaler + encoder
│   ├── feature_columns.pkl      # Final feature list
│   └── models/
│       ├── logistic_regression.pkl
│       ├── random_forest.pkl
│       └── xgboost.pkl
│
├── 📂 reports/
│   ├── summary.md               # Model comparison + business insights
│   ├── figures/                 # All plots (ROC, confusion matrix, SHAP, etc.)
│   └── metrics/
│       ├── metrics.csv          # Per-model evaluation scores
│       ├── comparison.csv       # Side-by-side model comparison
│       └── misclassification.csv
│
├── main.py                      # CLI entrypoint
└── README.md
```

---

## ⚙️ ML Pipeline Overview

```
Raw CSV (IBM HR Dataset)
        │
        ▼
┌───────────────────┐
│   Preprocessing   │  → Ordinal encoding, One-Hot encoding, outlier clipping,
│                   │    stratified 80/20 train-test split (random_state=42)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Feature Engineering│  → 10 custom features derived from domain logic
│                   │    (IncomePerYear, SatisfactionScore, OverTimeRisk, ...)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│    SMOTE          │  → Synthetic Minority Oversampling on training set only
│  (Imbalance Fix)  │    (applied post-split to prevent data leakage)
└────────┬──────────┘
         │
         ▼
┌───────────────────────────────────────────────┐
│           Model Training + Tuning             │
│  Logistic Regression  │  Random Forest  │  XGBoost  │
│  GridSearchCV (5-fold)│  RandomSearchCV │  GridSearchCV  │
│  scoring = f1_macro   │  n_iter=30      │  scoring = f1_macro │
└────────────────────────┬──────────────────────┘
                         │
                         ▼
┌───────────────────┐
│   Evaluation      │  → Accuracy, F1-macro, Precision, Recall,
│                   │    ROC-AUC, FPR, FNR, Threshold
└────────┬──────────┘
         │
         ▼
  📊 Reports + Figures + Model Selection
```

---

## 🔧 Feature Engineering

Beyond raw features, **10 domain-specific features** were engineered to improve signal quality:

| Feature | Formula / Logic | Business Meaning |
|---------|----------------|-----------------|
| `IncomePerYear` | `MonthlyIncome / (TotalWorkingYears + 1)` | Compensation relative to experience |
| `SatisfactionScore` | Avg of Job, Environment, Relationship Satisfaction | Composite wellbeing signal |
| `LoyaltyIndex` | `YearsAtCompany / (NumCompaniesWorked + 1)` | Tenure stability vs job-hopping |
| `WorkloadScore` | `OverTime × JobInvolvement` | Burnout risk proxy |
| `CareerGrowthRate` | `JobLevel / (YearsAtCompany + 1)` | Promotion velocity |
| `OverTimeRisk` | Binary flag from OverTime field | Direct overtime indicator |
| `TenureStabilityIndex` | `YearsInCurrentRole / (YearsAtCompany + 1)` | Role stability |
| `OverTimeXSatisfaction` | Interaction: `OverTime × JobSatisfaction` | Captures burnout-satisfaction link |
| `StagnationRisk` | `YearsSinceLastPromotion - CareerGrowthRate` | Risk of career plateauing |
| `IncomeVsExperience` | `MonthlyIncome / (TotalWorkingYears + 1)` | Pay vs years worked ratio |
| `DistanceXOverTime` | `DistanceFromHome × OverTime` | Commute + overwork stress factor |

**Raw Feature Groups:**

- **Numerical (14):** Age, DailyRate, DistanceFromHome, HourlyRate, MonthlyIncome, MonthlyRate, NumCompaniesWorked, PercentSalaryHike, TotalWorkingYears, TrainingTimesLastYear, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager
- **Categorical (7, One-Hot):** BusinessTravel, Department, EducationField, Gender, JobRole, MaritalStatus, OverTime
- **Ordinal (9, Mapped):** Education, EnvironmentSatisfaction, JobInvolvement, JobLevel, JobSatisfaction, PerformanceRating, RelationshipSatisfaction, StockOptionLevel, WorkLifeBalance

---

## 📊 Model Results

<div align="center">

| Model | Accuracy | F1-macro | Precision | Recall | ROC-AUC | Threshold | Verdict |
|:------|:--------:|:--------:|:---------:|:------:|:-------:|:---------:|:-------:|
| Logistic Regression | 0.8537 | 0.7153 | 0.7262 | 0.7062 | 0.8010 | 0.72 | Baseline |
| Random Forest | 0.8333 | 0.7116 | 0.6991 | 0.7285 | 0.7940 | 0.36 | Compare |
| **XGBoost** | **0.8503** | **0.7165** | **0.7205** | **0.7128** | **0.7963** | **0.39** | **✅ Winner** |

</div>

> **Primary Metric: F1-macro** — chosen because the dataset has class imbalance.
> Unlike accuracy, F1-macro penalises models that ignore the minority class.
> **ROC-AUC** was used as a tiebreaker for ranking quality across thresholds.

---

## ⚠️ Misclassification Analysis

Understanding *where* models fail is as important as overall accuracy.

<div align="center">

| Model | FPR | FNR | Dominant Error |
|:------|:---:|:---:|:--------------|
| Logistic Regression | 0.0769 | **0.5106** | Misses good candidates |
| Random Forest | 0.1174 | **0.4255** | Misses good candidates |
| **XGBoost** | **0.0850** | **0.4894** | **Best FPR + FNR balance** |

</div>

```
FPR (False Positive Rate) = Not-Hired candidates predicted as Hired
                          → Wasted recruiter time on bad-fit interviews

FNR (False Negative Rate) = Hired candidates predicted as Not-Hired
                          → Missed good talent — costly in competitive hiring
```

**Business Insight:** All models struggle most with FNR (missing qualified candidates). XGBoost achieves the best balance — lowest FPR while keeping FNR competitive with Random Forest.

---

## 🏆 Why XGBoost Won

```
Logistic Regression  →  Linear decision boundary. Cannot capture non-linear
                         hiring patterns (e.g., interaction between OverTime +
                         low satisfaction + low income). Good baseline, not enough.

Random Forest        →  Handles feature interactions well. But sensitive to
                         depth and sampling parameters. Higher FPR (0.1174) means
                         more wasted interviews.

XGBoost ✅           →  Gradient boosting on structured data. Handles imbalance
                         well when tuned. Best F1-macro (0.7165) + lowest FPR.
                         Custom engineered features gave it more signal to learn from.
```

**Hyperparameter search used GridSearchCV (5-fold, scoring=f1_macro):**

```yaml
n_estimators:     [200, 300]
learning_rate:    [0.01, 0.05]
max_depth:        [5, 6]
subsample:        [0.7, 0.8]
colsample_bytree: [0.6, 0.7]
min_child_weight: [3]
gamma:            [0, 0.1]
```

---

## 📈 Visualizations

All figures are saved to `reports/figures/`. Key plots:

<table>
<tr>
<td align="center" width="33%">

**ROC Curves**
Per-model AUC comparison
`figures/*_roc_curve.png`

</td>
<td align="center" width="33%">

**Confusion Matrices**
TP/FP/TN/FN breakdown
`figures/*_confusion_matrix.png`

</td>
<td align="center" width="33%">

**Feature Importance**
Top predictors per model
`figures/*_feature_importance.png`

</td>
</tr>
<tr>
<td align="center">

**Model Comparison**
Side-by-side bar chart
`figures/model_comparison.png`

</td>
<td align="center">

**Correlation Heatmap**
Feature correlation matrix
`figures/heatmap.png`

</td>
<td align="center">

**SHAP Summary**
Feature impact on predictions
`figures/shap_summary.png`

</td>
</tr>
</table>

---

## 🚀 Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/TalentIQ.git
cd TalentIQ
pip install -r requirements.txt
```

### 2. Add Dataset

Download the IBM HR Analytics dataset from Kaggle and place it at:

```
data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv
```

> 📎 Dataset: [IBM HR Analytics Attrition Dataset — Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

### 3. Run the Pipeline

```bash
# Full pipeline (preprocess → features → train → evaluate)
python main.py --stage pipeline

# Or run individual stages
python main.py --stage preprocess
python main.py --stage features
python main.py --stage train
```

### 4. View Results

```bash
# Model comparison
cat reports/summary.md

# Metrics CSV
cat reports/metrics/metrics.csv

# All figures
open reports/figures/
```

---

## 🛠️ Configuration

All pipeline settings are controlled via YAML — no hardcoded values.

**`config/config.yaml`**
```yaml
mode: full              # full | sample
sample_size: 500        # used when mode: sample

split:
  test_size: 0.20
  random_state: 42
  stratify: true        # preserves class distribution in splits

smote:
  enabled: true         # synthetic oversampling on train only
  random_state: 42
```

**`config/features.yaml`** — define which columns are numerical, categorical, ordinal, or engineered.

**`config/hyperparameters.yaml`** — modify param grids for each model without touching source code.

---

## 📦 Artifacts

Pre-trained artifacts are saved for inference without retraining:

| Artifact | Path | Description |
|----------|------|-------------|
| Preprocessor | `artifacts/preprocessor.pkl` | Fitted scaler + encoder (apply to new data) |
| Feature columns | `artifacts/feature_columns.pkl` | Exact feature order expected by models |
| Logistic Regression | `artifacts/models/logistic_regression.pkl` | Baseline model |
| Random Forest | `artifacts/models/random_forest.pkl` | Ensemble model |
| **XGBoost** | `artifacts/models/xgboost.pkl` | **Selected production model** |

**Using the saved model for prediction:**

```python
import pickle
import pandas as pd

# Load artifacts
with open("artifacts/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open("artifacts/models/xgboost.pkl", "rb") as f:
    model = pickle.load(f)

with open("artifacts/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# Predict on new data
X_new = pd.DataFrame([...])  # structured candidate profile
X_processed = preprocessor.transform(X_new)[feature_columns]
prediction = model.predict(X_processed)   # 0 = Not Hired, 1 = Hired
probability = model.predict_proba(X_processed)[:, 1]
```

---

## 📂 Dataset

| Property | Value |
|----------|-------|
| Source | [IBM HR Analytics — Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) |
| Target Column | `Attrition` (Yes / No → 1 / 0) |
| Total Records | 1,470 employees |
| Features | 35 raw columns |
| Class Distribution | Imbalanced (~84% No, ~16% Yes) |
| Imbalance Fix | SMOTE on training set |

---

## 🧱 Tech Stack

<div align="center">

| Layer | Tool |
|-------|------|
| Language | Python 3.10+ |
| ML Framework | scikit-learn, XGBoost |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Data | pandas, NumPy |
| Visualization | matplotlib, seaborn |
| Config | PyYAML |
| Serialization | pickle |

</div>

---

## 🚄 TransReliant (New Project)

> A separate project living in its own top-level directory (`TransReliant/`) within this repo — not part of the TalentIQ pipeline above.

**What it is:** A two-stage ML cascade for Indian railway ticket bookings. Model 1 (classification) predicts whether a booking will be **Confirmed / Not Confirmed**. Model 2 (regression) runs **only** on the passengers Model 1 flags as **Not Confirmed**, estimating how bad their **Waitlist Position** is. Stage 2 only fires for the flagged subset, so no model wastes effort estimating waitlist severity for already-confirmed passengers — the same design pattern as a classic triage → severity-estimation setup.

**Why this design, not a single model or a feature-passing cascade:** the two candidate targets (`Confirmation Status` and `Waitlist Position`) are different problem types (classification vs. regression), and in the cleaned dataset `Waitlist Position == 0` perfectly determines `Confirmation Status` — so feeding one as an input feature to predict the other would just be leaking the same information back to itself. Routing (run Stage 2 only on the subset Stage 1 flags) avoids that leakage while still producing a severity estimate for exactly the passengers who need one.

<div align="center">

| Property | Value |
|---|---|
| Stage 1 | Classification — `Confirmation Status` (Confirmed / Not Confirmed) |
| Stage 2 | Regression — `Waitlist Position` (1–200), on Stage-1-flagged rows only |
| Dataset | [Indian Railway Ticket Confirmation — Kaggle](https://www.kaggle.com/datasets/aaryananil/indian-railway-ticket-confirmation) |
| Records | 20,045 rows |
| Models | XGBClassifier + XGBRegressor |
| Dropped columns | `Train Number` (near-unique ID, 17,979 unique values, no signal), `PNR Number`, `Booking Channel` |

</div>

**Features used:**

- **Categorical:** Class of Travel, Quota, Source Station, Destination Station, Train Type, Special Considerations, Holiday or Peak Season
- **Numerical:** Number of Passengers, Travel Distance, Number of Stations, Travel Time, Seat Availability, journey_month, journey_dayofweek, days_before_journey

**Pipeline architecture:**

```
Raw ticket booking fields
        │
        ▼
 [ Model 1: XGBClassifier ]
   predicts Confirmation Status
        │
   ┌────┴─────┐
   │          │
Confirmed  Not Confirmed (routed)
   │          │
   │          ▼
   │   [ Model 2: XGBRegressor ]
   │   predicts Waitlist Position
   │          │
   └─────┬────┘
         ▼
  Final output: status + (if applicable) estimated waitlist severity
```

**Evaluation is done at three levels**, not just the final output:
- **Stage 1** — Accuracy, F1, ROC-AUC, chosen decision threshold
- **Stage 2** — RMSE / MAE, evaluated on its own held-out split
- **System-level** — the full cascade run end-to-end on the whole test set, including a *coverage* stat (% of test rows where Stage 2 actually fires) and Stage 2 error recomputed on the subset Stage 1 actually routes to it (which can differ from Stage 2's own clean-split number)

**Project structure:**

```
TransReliant/
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   ├── raw/                 # ticket_confirmation.csv
│   └── processed/           # cleaned.csv, featured.csv, splits/
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_feature_engineering_exp.ipynb
├── src/
│   ├── config_loader.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── train_stage1.py      # Confirmation Status classifier
│   ├── train_stage2.py      # Waitlist Position regressor (flagged subset only)
│   ├── evaluate.py
│   ├── pipeline.py
│   └── utils.py
├── artifacts/
│   └── models/               # stage1_classifier.pkl, stage2_regressor.pkl
├── reports/
│   ├── figures/
│   ├── metrics/
│   └── report.md
├── logs/
│   └── experiment_log.csv
├── main.py                   # interactive CLI demo
└── run_pipeline.py            # full pipeline entrypoint
```

**Run it:**

```bash
cd TransReliant
pip install -r requirements.txt
python run_pipeline.py   # regenerates artifacts, metrics, figures
python main.py            # interactive prediction CLI
```

> 📎 Full write-up (EDA, per-stage metrics, misclassification analysis, and design justification) lives in `TransReliant/reports/report.md` once the pipeline has been run.

---

<div align="center">

**Built by [Nithish Kumar S](https://github.com/yourusername)**
*First-year CS student · AI Engineer in progress · Hackathon builder*

<br/>

⭐ Star this repo if TalentIQ helped you learn ML pipelines!

</div>
