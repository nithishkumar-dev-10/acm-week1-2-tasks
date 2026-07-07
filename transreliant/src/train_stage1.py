# src/train_stage1.py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from config_loader import load_config
from preprocessing import build_stage1_preprocessor
from utils import log_experiment


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


def pick_best_model(results: dict) -> str:
    best_name = max(results, key=lambda name: results[name]["f1_mean"])
    print(f"Best model by mean CV F1: {best_name} (F1={results[best_name]['f1_mean']:.4f})")
    return best_name


def main():
    cfg = load_config()
    df = load_featured_data(cfg)
    X_train, X_test, y_train, y_test = split_and_save_stage1(df, cfg)
    results = compare_models(X_train, y_train, cfg)
    best_name = pick_best_model(results)
    print(f"\nStage 1 model comparison complete. '{best_name}' will be tuned next (Step 11).")


if __name__ == "__main__":
    main()