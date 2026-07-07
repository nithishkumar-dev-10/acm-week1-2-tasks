# src/train.py
"""
Stage 1 training pipeline — merges the former Step 10 (baseline model
comparison, train_stage1.py) and Step 11 (hyperparameter tuning,
tune_stage1.py) into a single script.

IMPORTANT CONTEXT (read before trusting the numbers this script prints):
Baseline comparison showed AUC ~0.50 for logreg/rf/xgb — i.e. no model
could rank Confirmed vs Not-Confirmed better than a coin flip. A raw-data
diagnostic (chi2 / correlation of every feature against Confirmation Status)
confirmed every usable feature is statistically unrelated to the target.
RF's high F1 in the baseline comparison was the class-imbalance trap: with
66.5% positive class, leaning toward "Confirmed" buys F1 without any real
discrimination.

So the tuning stage selects on ROC-AUC, not F1 — AUC can't be gamed by
class imbalance the way F1 can, so it's the honest metric here. Expect
tuning to produce only marginal AUC movement around 0.50. That is the
*correct* result given the data, not a bug in this script. If test AUC
comes back meaningfully above ~0.52-0.53, treat it with suspicion (recheck
for leakage) rather than celebrating it.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

from config_loader import load_config
from preprocessing import build_stage1_preprocessor
from utils import log_experiment

AAA_LOW_AUC_THRESHOLD = 0.55  # below this, we flag rather than declare victory


# ---------------------------------------------------------------------------
# Data loading / splitting
# ---------------------------------------------------------------------------

def load_featured_data(cfg: dict) -> pd.DataFrame:
    path = Path(cfg["data"]["featured"])
    if not path.exists():
        raise FileNotFoundError(f"Featured data not found at {path.resolve()} — run feature_engineering.py first.")
    df = pd.read_csv(path)
    print(f"Loaded featured data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def split_and_save_stage1(df: pd.DataFrame, cfg: dict):
    """
    Stratified 80/20 split on Confirmation Status (stratify because of the
    class imbalance from EDA). Persist all 4 pieces to disk so no script
    ever re-splits with a different seed downstream.
    """
    target_col = cfg["target"]["stage1"]
    numeric_cols = cfg["features"]["numerical"]
    categorical_cols = cfg["features"]["categorical"]
    feature_cols = numeric_cols + categorical_cols

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=cfg["random_seed"]
    )

    splits_dir = Path(cfg["data"]["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(splits_dir / "stage1_X_train.csv", index=False)
    X_test.to_csv(splits_dir / "stage1_X_test.csv", index=False)
    y_train.to_csv(splits_dir / "stage1_y_train.csv", index=False)
    y_test.to_csv(splits_dir / "stage1_y_test.csv", index=False)

    print(f"Stage 1 split saved: train={X_train.shape[0]} rows, test={X_test.shape[0]} rows "
          f"(train class balance: {y_train.mean():.3f} confirmed)")
    return X_train, X_test, y_train, y_test


def load_stage1_splits(cfg: dict):
    """Re-load the persisted splits (used when resuming straight into tuning)."""
    splits_dir = Path(cfg["data"]["splits_dir"])
    X_train = pd.read_csv(splits_dir / "stage1_X_train.csv")
    X_test = pd.read_csv(splits_dir / "stage1_X_test.csv")
    y_train = pd.read_csv(splits_dir / "stage1_y_train.csv").iloc[:, 0]
    y_test = pd.read_csv(splits_dir / "stage1_y_test.csv").iloc[:, 0]
    print(f"Loaded Stage 1 splits: train={X_train.shape[0]}, test={X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Step 10 — baseline model comparison
# ---------------------------------------------------------------------------

def compare_models(X_train, y_train, cfg: dict) -> dict:
    """
    5-fold StratifiedKFold CV across LogReg / RandomForest / XGBoost.
    Scored on F1 and ROC-AUC — not accuracy, because of class imbalance.
    Every model's CV result gets logged regardless of who wins.
    """
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos

    models = {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "rf": RandomForestClassifier(class_weight="balanced", random_state=cfg["random_seed"]),
        "xgb": XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=cfg["random_seed"],
        ),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg["random_seed"])
    results = {}

    for name, model in models.items():
        preprocessor = build_stage1_preprocessor(cfg)
        pipe = Pipeline([("prep", preprocessor), ("model", model)])

        f1_scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="f1")
        auc_scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="roc_auc")

        log_experiment(
            stage="1", model_name=name,
            cv_metric_name="f1", cv_metric_mean=f1_scores.mean(), cv_metric_std=f1_scores.std(),
        )
        log_experiment(
            stage="1", model_name=name,
            cv_metric_name="roc_auc", cv_metric_mean=auc_scores.mean(), cv_metric_std=auc_scores.std(),
        )

        results[name] = {"f1_mean": f1_scores.mean(), "f1_std": f1_scores.std(),
                          "auc_mean": auc_scores.mean(), "auc_std": auc_scores.std()}
        print(f"{name}: F1={f1_scores.mean():.4f} (+/-{f1_scores.std():.4f})  "
              f"AUC={auc_scores.mean():.4f} (+/-{auc_scores.std():.4f})")

    return results


def pick_best_baseline_model(results: dict) -> str:
    best_name = max(results, key=lambda name: results[name]["f1_mean"])
    print(f"Best model by mean CV F1: {best_name} (F1={results[best_name]['f1_mean']:.4f})")
    return best_name


# ---------------------------------------------------------------------------
# Step 11 — hyperparameter tuning
# ---------------------------------------------------------------------------

def get_search_spaces(cfg: dict, y_train: pd.Series) -> dict:
    """
    Modest grids — this dataset doesn't reward large search budgets (see
    module docstring), so RandomizedSearchCV with n_iter capped keeps this
    fast instead of burning time chasing noise.
    """
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos

    spaces = {
        "logreg": {
            "estimator": LogisticRegression(max_iter=2000, class_weight="balanced"),
            "params": {
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__penalty": ["l2"],
                "model__solver": ["lbfgs"],
            },
            "n_iter": 4,
        },
        "rf": {
            "estimator": RandomForestClassifier(class_weight="balanced", random_state=cfg["random_seed"]),
            "params": {
                "model__n_estimators": [200, 400, 600],
                "model__max_depth": [4, 8, 12, None],
                "model__min_samples_leaf": [1, 5, 20, 50],
                "model__max_features": ["sqrt", "log2"],
            },
            "n_iter": 10,
        },
        "xgb": {
            "estimator": XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=cfg["random_seed"],
            ),
            "params": {
                "model__n_estimators": [200, 400, 600],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__subsample": [0.7, 0.85, 1.0],
                "model__colsample_bytree": [0.7, 0.85, 1.0],
            },
            "n_iter": 10,
        },
    }
    return spaces


def tune_all_models(X_train, y_train, cfg: dict) -> dict:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg["random_seed"])
    spaces = get_search_spaces(cfg, y_train)
    results = {}

    for name, spec in spaces.items():
        preprocessor = build_stage1_preprocessor(cfg)
        pipe = Pipeline([("prep", preprocessor), ("model", spec["estimator"])])

        search = RandomizedSearchCV(
            pipe,
            param_distributions=spec["params"],
            n_iter=spec["n_iter"],
            scoring="roc_auc",
            cv=skf,
            random_state=cfg["random_seed"],
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)

        best_pipe = search.best_estimator_
        cv_auc_mean = search.best_score_
        cv_auc_std = search.cv_results_["std_test_score"][search.best_index_]

        # Also report F1 for the same tuned config, for transparency —
        # NOT used for selection (see module docstring on the F1 trap).
        f1_scores = cross_val_score(best_pipe, X_train, y_train, cv=skf, scoring="f1")

        log_experiment(
            stage="1", model_name=f"{name}_tuned",
            params=search.best_params_,
            cv_metric_name="roc_auc", cv_metric_mean=cv_auc_mean, cv_metric_std=cv_auc_std,
        )
        log_experiment(
            stage="1", model_name=f"{name}_tuned",
            params=search.best_params_,
            cv_metric_name="f1", cv_metric_mean=f1_scores.mean(), cv_metric_std=f1_scores.std(),
        )

        print(f"{name}: best CV AUC={cv_auc_mean:.4f} (+/-{cv_auc_std:.4f})  "
              f"F1={f1_scores.mean():.4f} (+/-{f1_scores.std():.4f})  "
              f"params={search.best_params_}")

        results[name] = {
            "pipeline": best_pipe,
            "cv_auc_mean": cv_auc_mean,
            "cv_auc_std": cv_auc_std,
            "cv_f1_mean": f1_scores.mean(),
            "cv_f1_std": f1_scores.std(),
            "best_params": search.best_params_,
        }

    return results


def pick_best_tuned_model(results: dict) -> str:
    best_name = max(results, key=lambda name: results[name]["cv_auc_mean"])
    print(f"\nBest tuned model by mean CV AUC: {best_name} "
          f"(AUC={results[best_name]['cv_auc_mean']:.4f})")
    return best_name


# ---------------------------------------------------------------------------
# Evaluation / artifact saving
# ---------------------------------------------------------------------------

def evaluate_on_test(best_pipe, X_test, y_test, model_name: str):
    y_pred = best_pipe.predict(X_test)
    y_proba = best_pipe.predict_proba(X_test)[:, 1]

    test_f1 = f1_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n=== Held-out test performance ({model_name}) ===")
    print(f"Test F1:  {test_f1:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print("Confusion matrix:\n", cm)
    print("\nClassification report:\n", classification_report(y_test, y_pred))

    log_experiment(
        stage="1", model_name=f"{model_name}_tuned_TEST",
        test_metric_name="roc_auc", test_metric_value=test_auc,
    )
    log_experiment(
        stage="1", model_name=f"{model_name}_tuned_TEST",
        test_metric_name="f1", test_metric_value=test_f1,
    )

    if test_auc < AAA_LOW_AUC_THRESHOLD:
        print(
            "\n" + "!" * 78 +
            f"\nDIAGNOSTIC: test AUC ({test_auc:.4f}) is at/near random chance (0.50).\n"
            "This matches the raw-data audit: no available feature is statistically\n"
            "related to Confirmation Status. Tuning cannot manufacture signal that\n"
            "isn't in the data. This is a DATASET finding, not a modeling failure —\n"
            "document it as such rather than continuing to tune Stage 1 further.\n"
            "Recommended next step: revisit the raw data source (or discuss with\n"
            "your mentor whether this dataset is fit for a Stage 1 classifier at all)\n"
            "before sinking more time into Step 12+.\n" + "!" * 78
        )

    return test_f1, test_auc


def save_artifacts(best_pipe, cfg: dict):
    prep_path = Path(cfg["artifacts"]["preprocessor_stage1"])
    model_path = Path(cfg["artifacts"]["model_stage1"])
    prep_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_pipe.named_steps["prep"], prep_path)
    joblib.dump(best_pipe.named_steps["model"], model_path)
    print(f"\nSaved tuned preprocessor -> {prep_path}")
    print(f"Saved tuned model        -> {model_path}")


# ---------------------------------------------------------------------------
# Entry point — baseline comparison -> tuning -> evaluation -> save
# ---------------------------------------------------------------------------

def main():
    cfg = load_config()

    # Step 10: load data, split, baseline model comparison
    df = load_featured_data(cfg)
    X_train, X_test, y_train, y_test = split_and_save_stage1(df, cfg)
    baseline_results = compare_models(X_train, y_train, cfg)
    baseline_best_name = pick_best_baseline_model(baseline_results)
    print(f"\nStage 1 model comparison complete. '{baseline_best_name}' will be tuned next.")

    # Step 11: hyperparameter tuning, selected on CV AUC (not F1 — see module docstring)
    tuning_results = tune_all_models(X_train, y_train, cfg)
    best_name = pick_best_tuned_model(tuning_results)
    best_pipe = tuning_results[best_name]["pipeline"]

    evaluate_on_test(best_pipe, X_test, y_test, best_name)
    save_artifacts(best_pipe, cfg)

    print(f"\nStage 1 training complete. Tuned '{best_name}' selected by CV AUC and saved as Stage 1 artifact.")


if __name__ == "__main__":
    main()