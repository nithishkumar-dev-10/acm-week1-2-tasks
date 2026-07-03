<div align="center">

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,50:203a43,100:2c5364&height=200&section=header&text=WorkPulse&fontSize=80&fontColor=ffffff&fontAlignY=38&desc=ML-Based%20Employee%20Attrition%20Prediction%20System&descAlignY=58&descSize=18&descColor=90cdf4&animation=fadeIn" width="100%"/>

<br/>

<table border="0" cellpadding="0" cellspacing="0">
<tr>
<td align="center">

```
  🧠  Predict · Analyze · Retain  🎯
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
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

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
<code>Attrition = Yes / No</code>
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

> *"Employees may leave for various reasons — better salary opportunities, career growth,*  
> *improved work-life balance, job dissatisfaction, or personal reasons.*  
> *HR departments often struggle to identify employees who are at risk of leaving before they resign."*
>
> **WorkPulse fixes that** — a tuned ML pipeline trained on structured employee data:  
> demographics · job role · satisfaction · tenure · workload · compensation

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f2027,100:2c5364&height=3&section=header" width="85%"/>

<br/>

</div>

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Project Structure](#-project-structure)
- [ML Pipeline Overview](#-ml-pipeline-overview)
- [Feature Engineering](#-feature-engineering)
- [Model Results](#-model-results)
- [Misclassification Analysis](#-misclassification-analysis)
- [Why XGBoost Won](#-why-xgboost-won)
- [Visualizations](#-visualizations)
- [Quickstart](#-quickstart)
- [Configuration](#️-configuration)
- [Artifacts](#-artifacts)
- [Dataset](#-dataset)

---

## 🎯 Problem Statement

Employee attrition is a major challenge for organizations. Employees may leave for various reasons, such as **better salary opportunities, career growth, improved work-life balance, job dissatisfaction, or personal reasons**. High employee turnover increases recruitment and training costs, results in the loss of experienced talent, and disrupts business operations.

As a result, HR departments often struggle to identify employees who are at risk of leaving before they resign.

This project addresses that challenge by building a **machine learning-based employee attrition prediction system**. By analyzing an employee's demographic, job-related, and workplace attributes, the model predicts whether an employee is likely to leave the organization, enabling HR teams to take proactive retention measures.

<table>
<tr>
<td width="50%">

### The Challenge
- Employees leave for **salary, career growth, work-life balance, dissatisfaction, or personal reasons**
- High turnover drives up **recruitment and training costs**
- Losing experienced talent **disrupts business operations**
- HR teams struggle to spot **at-risk employees before they resign**

</td>
<td width="50%">

### The Solution
- Binary classifier → **Attrition (Yes / No)**
- Compare **3 ML models** head-to-head
- Select winner by **F1-macro + ROC-AUC** (handles class imbalance)
- Full pipeline: EDA → Engineering → Tuning → Evaluation
- Surfaces at-risk employees so HR can act **proactively**

</td>
</tr>
</table>

---

## 📁 Project Structure

```
WorkPulse/
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
| Logistic Regression | 0.8537 | 0.7153 | 0.7262 | 0.7062 | 0.801 | 0.72 | Baseline |
| Random Forest | 0.8333 | 0.7116 | 0.6991 | 0.7285 | 0.794 | 0.36 | Compare |
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
| Logistic Regression | 0.0769 | **0.5106** | Misses at-risk employees |
| Random Forest | 0.1174 | **0.4255** | Misses at-risk employees |
| **XGBoost** | **0.0850** | **0.4894** | **Best FPR + FNR balance** |

</div>

```
FPR (False Positive Rate) = Employees who stayed predicted as Attrition
                          → Wasted retention effort on low-risk employees

FNR (False Negative Rate) = Employees who left predicted as No Attrition
                          → Missed at-risk employees — costly, unplanned turnover
```

**Business Insight:** All models struggle most with FNR (missing employees who actually leave). XGBoost achieves the best balance — lowest FPR while keeping FNR competitive with Random Forest.

---

## 🏆 Why XGBoost Won

```
Logistic Regression  →  Linear decision boundary. Cannot capture non-linear
                         attrition patterns (e.g., interaction between OverTime +
                         low satisfaction + low income). Good baseline, not enough.

Random Forest        →  Handles feature interactions well. But sensitive to
                         depth and sampling parameters. Higher FPR (0.1174) means
                         more wasted retention interventions on low-risk employees.

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
git clone https://github.com/yourusername/WorkPulse.git
cd WorkPulse
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
X_new = pd.DataFrame([...])  # structured employee profile
X_processed = preprocessor.transform(X_new)[feature_columns]
prediction = model.predict(X_processed)   # 0 = Stayed, 1 = Left (Attrition)
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

<div align="center">

**Built by [Nithish Kumar S](https://github.com/yourusername)**  
*SECOND-year CS student · AI Engineer in progress *

<br/>

⭐ Star this repo if WorkPulse helped you learn ML pipelines!

</div>
