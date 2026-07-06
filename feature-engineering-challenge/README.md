<div align="center">

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,50:203a43,100:2c5364&height=200&section=header&text=Feature%20Engineering%20Challenge&fontSize=60&fontColor=ffffff&fontAlignY=38&desc=Santander%20Customer%20Transaction%20Prediction&descAlignY=58&descSize=18&descColor=90cdf4&animation=fadeIn" width="100%"/>

<br/>

<table border="0" cellpadding="0" cellspacing="0">
<tr>
<td align="center">

```
  🎯  Engineer · Diagnose · Tune  🧠
```

</td>
</tr>
</table>

<br/>

<!-- STAT PILLS -->
<img src="https://img.shields.io/badge/Recall-88.79%25-0ea5e9?style=flat-square&labelColor=0f172a" />
&nbsp;
<img src="https://img.shields.io/badge/Precision-21.42%25-8b5cf6?style=flat-square&labelColor=0f172a" />
&nbsp;
<img src="https://img.shields.io/badge/F1-0.3452-10b981?style=flat-square&labelColor=0f172a" />
&nbsp;
<img src="https://img.shields.io/badge/Model-Logistic%20Regression-f97316?style=flat-square&labelColor=0f172a" />
&nbsp;
<img src="https://img.shields.io/badge/Experiments-10%20Logged-ec4899?style=flat-square&labelColor=0f172a" />

<br/><br/>

<!-- TECH STACK BADGES -->
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Logistic%20Regression-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![pandas](https://img.shields.io/badge/pandas-Data%20Wrangling-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)

<br/>

[![Dataset](https://img.shields.io/badge/📊%20Dataset-Santander%20Transaction%20Prediction-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/santander-customer-transaction-prediction)
[![Constraint](https://img.shields.io/badge/Constraint-LR%20Only%2C%20No%20External%20Data-22C55E?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-MIT-a855f7?style=for-the-badge)](LICENSE)

<br/>

---

<table border="0" width="85%">
<tr>
<td align="center" width="25%" style="padding: 10px">
<br/>
<b>🎯 Task</b><br/>
Binary Classification<br/>
<code>target = 1 / 0</code>
</td>
<td align="center" width="25%" style="padding: 10px">
<br/>
<b>📦 Dataset</b><br/>
Santander (Kaggle)<br/>
<code>200 anonymized features</code>
</td>
<td align="center" width="25%" style="padding: 10px">
<br/>
<b>🔬 Model</b><br/>
Logistic Regression<br/>
<code>Only model family allowed</code>
</td>
<td align="center" width="25%" style="padding: 10px">
<br/>
<b>✅ Target Hit</b><br/>
Recall 0.83 → 0.888<br/>
<code>+ threshold tuning</code>
</td>
</tr>
</table>

<br/>

> *"The baseline model achieved 0.83 recall — good, but not good enough to reliably*  
> *catch the customers who actually go on to transact.*  
> *Model swaps were off the table. The only lever left was the data itself."*
>
> **RecallForge fixes that** — a fully feature-engineered pipeline built strictly on top of Logistic Regression:  
> frequency signals · row-level statistics · interactions · redundancy pruning · decision-boundary tuning

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f2027,100:2c5364&height=3&section=header" width="85%"/>

<br/>

</div>

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Project Structure](#-project-structure)
- [ML Pipeline Overview](#️-ml-pipeline-overview)
- [Feature Engineering](#-feature-engineering)
- [Experiment Results](#-experiment-results)
- [The Fake-Recall Trap](#-the-fake-recall-trap)
- [Why Custom Weighting Won](#-why-custom-weighting-won)
- [Quickstart](#-quickstart)
- [Configuration](#️-configuration)
- [Artifacts](#-artifacts)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)

---

## 🎯 Problem Statement

The Santander Customer Transaction Prediction dataset is heavily imbalanced — roughly **90% class 0, 10% class 1** — with 200 anonymized numeric columns that individually correlate very weakly with the target (strongest single-column correlation ≈ 0.08). A baseline Logistic Regression model reaches **0.83 recall**, but the assignment requires pushing this to **0.88+** using **feature engineering alone** — no other model family, no external data.

<table>
<tr>
<td width="50%">

### The Challenge
- Class imbalance (~90/10) makes recall hard to move without hurting precision
- Individual columns carry almost no signal on their own
- Only Logistic Regression is allowed — no ensembles, no boosting
- No external data — every feature must come from the given 200 columns

</td>
<td width="50%">

### The Solution
- Engineer signal-dense features: frequency, row stats, interactions
- Prune redundant/near-constant columns before tuning
- Diagnose "fake" recall gains caused by naive class weighting
- Replace blunt `class_weight='balanced'` with a tuned weight + threshold combo

</td>
</tr>
</table>

---

## 📁 Project Structure

```
feat/
│
├── 📂 notebooks/
│   └── santander_feature_engineering.ipynb   # Full pipeline, stage by stage
│
├── 📂 logs/
│   └── experiment_log.csv                    # Every experiment, in order, with scores
│
├── 📂 reports/
│   └── report.md                             # Write-up: what worked, what didn't, why
│
├── 📂 artifacts/
│   └── final_model.pkl                       # Model + scaler + threshold + feature list
│
├── requirements.txt                          # Pinned dependencies
├── .env.sample                               # Template for your Kaggle API token
└── README.md
```

---

## ⚙️ ML Pipeline Overview

```
Raw CSV (Santander Transaction Data, via kagglehub)
        │
        ▼
┌───────────────────┐
│  Train/Val/Test    │  → 85/15 split BEFORE any feature engineering
│      Split         │    (prevents leakage into engineered features)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ True Baseline LR   │  → Scaled, class_weight='balanced'
│                   │    Recall 0.7713 · F1 0.4149 (working baseline)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Frequency Encoding │  → One column per feature: how common is this value?
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Row-Wise Stats    │  → mean / std / min / max / skew across all 200 cols
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   Interactions     │  → Pairwise products of top-20 correlated columns
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Redundancy Pruning │  → Drop near-constant / highly-correlated columns
│                   │    (595 → 509 features)
└────────┬──────────┘
         │
         ▼
┌───────────────────────────────────────────────┐
│         Tuning: C · penalty · class_weight     │
│   Grid search  →  custom weight + threshold sweep │
└────────────────────────┬──────────────────────┘
                         │
                         ▼
  📊  Final Model  →  Test Recall 0.8879 (evaluated once)
```

---

## 🔧 Feature Engineering

| Feature Type | Logic | Why It Helps |
|---|---|---|
| **Frequency Encoding** | New column per feature = how common that value is in train | Raw columns barely correlate with target; frequency gives the model a different signal entirely |
| **Row-Wise Statistics** | `mean`, `std`, `min`, `max`, `skew` across all 200 columns, per row | Gives the model a "summary view" of each customer no single column can provide |
| **Interaction Features** | Pairwise products of the top-20 most correlated columns | Captures combined effects without the noise of all-pairs on 200 columns |
| **Outlier Capping** | 1st/99th percentile clipping on original columns | Tested for robustness — see [results](#-experiment-results) for why it was dropped |
| **Redundancy Pruning** | Drop near-constant columns + highly-correlated pairs | Cuts 595 → 509 features with no F1 loss, keeps the model leaner |

All statistics were fit on the **training split only** and applied unchanged to validation/test — never the other way around.

---

## 📊 Experiment Results

<div align="center">

| Stage | Recall | Precision | F1 |
|:------|:------:|:---------:|:--:|
| Baseline (scaled, `class_weight='balanced'`) | 0.7713 | 0.2837 | 0.4149 |
| + Frequency encoding (no class_weight) | 0.8041 | 0.2642 | 0.3977 |
| + Row-wise stats | 0.6733 | 0.3829 | 0.4882 |
| + Interactions | 0.6807 | 0.3849 | **0.4918** |
| + Outlier capping | 0.6753 | 0.3833 | 0.4890 |
| + Redundancy pruning | 0.6760 | 0.3845 | 0.4902 |
| + Grid-tuned (`class_weight='balanced'`) | 0.9465 | 0.1618 | 0.2763 |
| + Custom weight + threshold sweep | 0.8837 | 0.2139 | 0.3444 |
| **Final — Test Set (evaluated once)** | **0.8879** | **0.2142** | **0.3452** |

</div>

Full run-by-run detail — including every dead end — lives in [`logs/experiment_log.csv`](logs/experiment_log.csv).

---

## 🕵️ The Fake-Recall Trap

Combining frequency encoding with `class_weight='balanced'` produced a recall of **0.9863** — which looked spectacular, until precision collapsed to **0.1256**.

```
Predicted class distribution:  [5,382  |  20,118]
Actual class distribution:     [22,938 |  2,562]

→ Model predicted class 1 for ~79% of rows when only ~10% actually are class 1.
  The "high recall" was fake — the model wasn't learning signal, it was just
  over-predicting the positive class.
```

**Business Insight:** Removing `class_weight='balanced'` (keeping the frequency features) fixed the degenerate behavior and confirmed the class weighting — not the features — caused the distortion. This diagnosis directly motivated replacing the blunt `'balanced'` switch with a tunable weight dial later in the pipeline.

---

## 🏆 Why Custom Weighting Won

```
class_weight=None        →  Recall 0.28-0.80. Model has no incentive to catch
                             the minority class at all. Far below target.

class_weight='balanced'  →  Recall 0.95, but overshoots hard. Precision
                             collapses to 0.16. Blunt on/off switch, no control
                             over how far past the recall target it goes.

Custom {0:1, 1:2.5} ✅   →  Acts as a DIAL instead of a switch. Combined with
   + threshold=0.40         a tuned decision threshold, lands right at the
                             0.88 recall requirement — with meaningfully
                             better precision and F1 than 'balanced'.
```

**Tuning approach — custom weight + threshold sweep:**

```python
weight_ratios = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9]
thresholds    = [0.10 → 0.50]

# Selection rule: filter to recall >= 0.88, then maximize F1
# Best found: weight_ratio=2.5, threshold=0.40
```

---

## 🚀 Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Kaggle Access

```bash
cp .env.sample .env
```

Fill in your Kaggle API token in `.env`. The notebook pulls the dataset via `kagglehub` straight into its own cache — this avoids storing large CSVs in the repo.

### 3. Run the Pipeline

Open and run top-to-bottom:

```
notebooks/santander_feature_engineering.ipynb
```

Every experiment is logged automatically to `logs/experiment_log.csv` as the notebook runs.

### 4. View Results

```bash
# Full write-up: what worked, what didn't, why
cat reports/report.md

# Every experiment, in order, with scores
cat logs/experiment_log.csv
```

---

## 🛠️ Configuration

| Setting | Value | Notes |
|---|---|---|
| Train/Val/Test split | 85% / — / 15% | Split before any feature engineering, `random_state=42` |
| Model family | `LogisticRegression` only | No other model family used at any stage |
| Final `class_weight` | `{0: 1, 1: 2.5}` | Found via weight + threshold sweep, not `'balanced'` |
| Final decision threshold | `0.40` | Applied to predicted probabilities, not the default 0.5 |
| External data | None | All features derived from the 200 given columns |

---

## 📦 Artifacts

| Artifact | Path | Description |
|---|---|---|
| Final model bundle | `artifacts/final_model.pkl` | Dict containing the fitted model, scaler, decision threshold, dropped-column list, and top-feature list |

**Using the saved model for prediction:**

```python
import joblib

bundle = joblib.load("artifacts/final_model.pkl")
model, scaler, threshold = bundle["model"], bundle["scaler"], bundle["threshold"]

X_scaled = scaler.transform(X_new)                       # X_new = engineered feature set
proba = model.predict_proba(X_scaled)[:, 1]
prediction = (proba >= threshold).astype(int)            # 1 = predicted transaction, 0 = not
```

---

## 📂 Dataset

| Property | Value |
|---|---|
| Source | [Santander Customer Transaction Prediction — Kaggle](https://www.kaggle.com/competitions/santander-customer-transaction-prediction) |
| Target Column | `target` (1 / 0) |
| Features | 200 anonymized numeric columns (`var_0`...`var_199`) |
| Class Distribution | Imbalanced (~90% class 0, ~10% class 1) |
| Imbalance Fix | Custom `class_weight` dial + tuned decision threshold |

---

## 🧱 Tech Stack

<div align="center">

| Layer | Tool |
|---|---|
| Language | Python 3.10+ |
| ML Framework | scikit-learn (Logistic Regression only) |
| Data | pandas, NumPy |
| Stats | SciPy (skew) |
| Data Access | kagglehub, python-dotenv |
| Visualization | matplotlib, seaborn |
| Serialization | joblib |

</div>

---

<div align="center">

**Feature Engineering Challenge — Individual Task Submission**  
*Only Logistic Regression · No external data · Test set touched once*

<br/>

⭐ Recall 0.83 → 0.888, through feature engineering alone.

</div>
