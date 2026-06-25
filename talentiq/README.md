# TalentIQ 🎯
> Binary ML classifier that predicts candidate shortlisting from structured resume data.

---

## Project Phases

| Phase | Focus | Status |
|-------|-------|--------|
| **Phase 1** | Setup — venv, configs, folder structure, dataset | ✅ Done |
| **Phase 2** | EDA + Preprocessing + Feature Engineering | 🔜 Next |
| **Phase 3** | Model Training + Hyperparameter Tuning | ⏳ Upcoming |
| **Phase 4** | Evaluation + SHAP + Final Report | ⏳ Upcoming |

---

## Phase 1 Setup Guide

### 1. Clone & enter project
```bash
git clone <your-repo-url>
cd talentiq
```

### 2. Create virtual environment
```bash
python -m venv venv

# Activate
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download dataset from Kaggle

First, get your Kaggle API token:
- Go to https://www.kaggle.com/settings → API → **Create New Token**
- This downloads `kaggle.json`

Place it:
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Then run:
```bash
pip install kaggle
python download_data.py
```

### 5. Verify everything works
```bash
python verify_setup.py
```
You should see **7/7 checks passed** ✅

---

## Folder Structure

```
talentiq/
├── data/
│   ├── raw/               ← original Kaggle CSV
│   ├── processed/         ← cleaned data (Phase 2)
│   └── splits/            ← train.csv / test.csv (Phase 2)
├── config/
│   ├── config.yaml        ← paths, seeds, imbalance strategy
│   ├── features.yaml      ← feature lists + ordinal maps
│   └── hyperparameters.yaml ← search spaces for all 3 models
├── src/
│   ├── config_loader.py   ← YAML loader utility
│   ├── utils.py           ← logger, seed, path helpers
│   ├── preprocessing.py   ← (Phase 2)
│   ├── feature_engineering.py ← (Phase 2)
│   ├── train.py           ← (Phase 3)
│   ├── predict.py         ← (Phase 3)
│   ├── metrics.py         ← (Phase 4)
│   └── plots.py           ← (Phase 4)
├── artifacts/
│   └── models/            ← saved .pkl model files
├── reports/
│   ├── figures/           ← EDA + evaluation plots
│   ├── metrics/           ← JSON metric dumps
│   └── summary.md         ← final model comparison
├── notebooks/             ← Jupyter notebooks
├── logs/pipeline.log
├── main.py
├── download_data.py
├── verify_setup.py
└── requirements.txt
```

---

## Dataset
**Resume Screening Dataset** — Kaggle (~200K candidates)  
Target: `Hired` (1 = shortlisted, 0 = rejected)

---

## Models Used
- Logistic Regression (GridSearchCV)
- Random Forest (RandomizedSearchCV)
- XGBoost (RandomizedSearchCV)

Primary metric: **F1-macro** | Secondary: **ROC-AUC**
