<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,50:203a43,100:2c5364&height=200&section=header&text=ACM%20SIG-AI%20TASKS&fontSize=60&fontColor=ffffff&fontAlignY=38&desc=Recruitment%20Task%20Portfolio%20%E2%80%94%20Nithish%20Kumar%20S&descAlignY=58&descSize=18&descColor=90cdf4&animation=fadeIn" width="100%"/>

<br/>

```
  🧠  Learn · Build · Ship  🚀
```

<img src="https://img.shields.io/badge/Projects-3-0ea5e9?style=flat-square&labelColor=0f172a" />
&nbsp;
<img src="https://img.shields.io/badge/Language-Python-8b5cf6?style=flat-square&labelColor=0f172a" />
&nbsp;
<img src="https://img.shields.io/badge/Focus-Applied%20ML-10b981?style=flat-square&labelColor=0f172a" />
&nbsp;
<img src="https://img.shields.io/badge/Status-In%20Progress-f97316?style=flat-square&labelColor=0f172a" />

</div>

---

## 👋 Introduction

This repository is a single home for every task completed as part of the **ACM SIG-AI recruitment process**. Each project was built as a stage-wise checkpoint — starting from core feature engineering, moving into a full classification pipeline, and finishing with a multi-stage, end-to-end cascade system.

Rather than three disconnected assignments, the projects are meant to be read **in sequence** — each one builds directly on the skills and mistakes of the one before it.

---

## 🧭 Learning Journey

```
Feature Engineering Challenge
        ↓
     WorkPulse
        ↓
TransReliant Cascade
```

| Stage | What it added |
|---|---|
| **Feature Engineering Challenge** | Learned to extract signal from weak, anonymized data using engineered features alone |
| **WorkPulse** | Learned to compare multiple models, handle class imbalance, and explain predictions |
| **TransReliant Cascade** | Learning to design a **multi-model system** — classification feeding into regression, with proper leakage control |

Each stage intentionally increases in scope: single-model tuning → model comparison → multi-stage pipeline architecture.

---

## 📦 Repository Overview

| Project Name | Stage | Status | Primary Focus |
|---|:---:|:---:|---|
| **Feature Engineering Challenge** | Week 1–2 | ✅ Done | Feature engineering under a fixed model constraint (Logistic Regression only) |
| **WorkPulse** | Week 3–4 | ✅ Done | Model comparison, tuning, and explainability (SHAP) on imbalanced data |
| **TransReliant Cascade** | Final Project | ✅ Done | Two-stage ML system: Classification → Regression cascade |

---

## 🗓️ Project Timeline

```
Week 1-2   →  Feature Engineering Challenge   (Santander Transaction Prediction)
Week 3-4   →  WorkPulse                       (IBM HR Attrition Prediction)
Final      →  TransReliant Cascade            (Indian Railway Ticket Confirmation)
```

---

## 1️⃣ Feature Engineering Challenge

**Dataset:** Santander Customer Transaction Prediction (Kaggle)

A binary classification task with one hard constraint: **only Logistic Regression is allowed** — no model swapping, no external data. The goal was to push recall from a baseline of **0.83 to 0.88+** purely through feature engineering.

**Key ideas explored:**
- Frequency encoding, row-wise statistics, and pairwise interaction features to extract signal from 200 anonymized, weakly-correlated columns
- Diagnosing a **"fake recall" trap**, where naive `class_weight='balanced'` inflated recall by over-predicting the positive class
- Replacing a blunt class-weight switch with a **tuned weight + decision-threshold dial**

**Result:** Recall of **0.888** with a defensible precision/F1 trade-off, using feature engineering alone.

---

## 2️⃣ WorkPulse

**Dataset:** IBM HR Analytics Attrition Dataset (Kaggle)

An ML-based employee attrition prediction system built to help HR teams flag at-risk employees before they resign.

**Key ideas explored:**
- Comparing **Logistic Regression, Random Forest, and XGBoost** head-to-head via cross-validated tuning
- **10 domain-driven engineered features** (e.g. `WorkloadScore`, `LoyaltyIndex`, `StagnationRisk`) to capture burnout and turnover risk
- Handling class imbalance with **SMOTE** applied strictly on the training split
- Model explainability via **SHAP**, and error analysis via False Positive/Negative rates

**Result:** XGBoost selected as the winning model (**F1-macro 0.7165, ROC-AUC 0.7963**) for the best balance of missed-attrition risk vs. wasted retention effort.

---

## 3️⃣ TransReliant Cascade

**Dataset:** Indian Railway Ticket Confirmation Dataset (Kaggle)

A two-stage cascade system designed to predict ticket confirmation status, and — for passengers flagged as **not confirmed** — estimate how severe their waitlist position is likely to be.

**Planned architecture:**
- **Model 1 (Classification):** Predicts `Confirmation Status` from booking and journey features
- **Model 2 (Regression):** Runs only on the subset flagged "Not Confirmed" by Model 1, predicting `Waitlist Position`
- Genuine cascade design: Model 2 is trained on ground-truth labels but **routed** by Model 1's predictions at inference time — avoiding compounded errors and label leakage
- Full pipeline planned: EDA → cleaning → feature engineering (`seat_pressure`, `booking_urgency_bucket`) → dual `ColumnTransformer` preprocessing → per-stage model comparison and tuning → threshold optimization → system-level evaluation

**Key features planned:** CLI demo, pipeline architecture diagram, experiment logging, and a full written report covering per-stage and system-wide evaluation.

---

## 🗂️ Repository Structure

```
ACM-SIG-AI-TASKS/
│
├── 01-feature-engineering-challenge/
│   ├── notebooks/
│   ├── logs/
│   ├── reports/
│   ├── artifacts/
│   └── README.md
│
├── 02-workpulse/
│   ├── config/
│   ├── src/
│   ├── notebooks/
│   ├── data/
│   ├── artifacts/
│   ├── reports/
│   └── README.md
│
├── 03-transreliant-cascade/
│   ├── data/
│   ├── src/
│   ├── notebooks/
│   ├── artifacts/
│   ├── reports/
│   └── README.md
│
└── README.md          ← you are here
```

---

## 🧱 Technologies Used

<div align="center">

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| ML Frameworks | scikit-learn, XGBoost |
| Imbalance Handling | imbalanced-learn (SMOTE, class weighting) |
| Explainability | SHAP, permutation importance |
| Data | pandas, NumPy |
| Visualization | matplotlib, seaborn |
| Config Management | PyYAML |
| Serialization | joblib, pickle |
| Data Access | kagglehub, python-dotenv |

</div>

---

## 🎓 Skills Covered

- Feature engineering under model constraints
- Handling class imbalance (SMOTE, custom class weights, threshold tuning)
- Multi-model comparison and hyperparameter tuning (GridSearch / RandomizedSearch)
- Model explainability (SHAP, feature importance)
- Leakage-aware data pipeline design
- Multi-stage cascade architecture (classification → regression)
- Config-driven, reproducible ML pipelines
- Experiment logging and structured evaluation reporting

---

## 📌 Current Repository Status

| Project | Status |
|---|:---:|
| Feature Engineering Challenge | ✅ Done |
| WorkPulse | ✅ Done |
| TransReliant Cascade | ✅ Done |

---

## 🙌 Closing Note

This repository reflects a deliberate, stage-by-stage progression — from squeezing signal out of raw features, to comparing and explaining models, to architecting a complete multi-model system. Each project builds on the lessons of the last, with the long-term goal of designing and shipping real, production-minded AI systems.

<div align="center">

**Nithish Kumar S** · B.Tech CS · Building toward AI Engineering

</div>
