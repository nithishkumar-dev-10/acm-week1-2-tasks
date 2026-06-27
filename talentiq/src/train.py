import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    StandardScaler,
)
from xgboost import XGBClassifier

from src.config_loader import PROJECT_ROOT, load_config, load_features, load_hyperparameters
from src.feature_engineering import engineer_features
from src.metrics import evaluate_model, save_metrics, save_summary
from src.plots import plot_confusion_matrix, plot_feature_importance, plot_model_comparison, plot_roc_curve
from src.preprocessing import preprocess_data

logger = logging.getLogger(__name__)


# ── 1. BUILD PREPROCESSOR PIPELINE ──────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """
    ColumnTransformer that handles all encoding and scaling.
    Fitted on train only — applied to both train and test.
    """
    feat = load_features()

    # ordinal columns have a natural order defined in features.yaml
    ordinal_cols = list(feat["ordinal_maps"].keys())          # ['education_level', 'university_tier']
    ordinal_categories = [
        list(feat["ordinal_maps"][col].keys())                # e.g. ['High School','Bachelor','Master','PhD']
        for col in ordinal_cols
    ]

    # one-hot columns have no natural order
    onehot_cols = feat["onehot_columns"]                      # ['company_type']

    # numerical columns get scaled
    numerical_cols = feat["numerical_features"]               # 12 raw + 7 engineered added later

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "ordinal",
                OrdinalEncoder(
                    categories=ordinal_categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ordinal_cols,
            ),
            (
                "onehot",
                OneHotEncoder(
                    drop="first",
                    sparse_output=False,
                    handle_unknown="ignore",
                ),
                onehot_cols,
            ),
            (
                "scaler",
                StandardScaler(),
                numerical_cols,
            ),
        ],
        remainder="passthrough",   # engineered features pass through untouched
    )

    return preprocessor


# ── 2. LOAD AND PREPARE DATA ─────────────────────────────────────────────────

def load_and_prepare() -> tuple:
    """
    Run preprocessing + feature engineering, then split into
    X_train, X_test, y_train, y_test. Saves splits to data/splits/.
    """
    cfg  = load_config()
    feat = load_features()

    df = preprocess_data()
    df = engineer_features(df)

    target = feat["target"]
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["random_state"],
        stratify=y if cfg["split"]["stratify"] else None,
    )

    # save raw splits for reference / notebook inspection
    splits_dir = PROJECT_ROOT / "data" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(splits_dir / "X_train.csv", index=False)
    X_test.to_csv(splits_dir  / "X_test.csv",  index=False)
    y_train.to_csv(splits_dir / "y_train.csv", index=False)
    y_test.to_csv(splits_dir  / "y_test.csv",  index=False)

    logger.info(
        f"Split — Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows"
    )

    return X_train, X_test, y_train, y_test


# ── 3. APPLY PREPROCESSOR + SMOTE ────────────────────────────────────────────

def transform_and_resample(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
) -> tuple:
    """
    Fit preprocessor on train only, transform both.
    Apply SMOTE on transformed train only (prevents data leakage).
    """
    cfg = load_config()

    # fit on train, transform both
    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc  = preprocessor.transform(X_test)

    # save fitted preprocessor
    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, artifacts_dir / "preprocessor.pkl")
    logger.info("Preprocessor saved to artifacts/preprocessor.pkl")

    # save feature column list
    feature_cols = list(X_train.columns)
    joblib.dump(feature_cols, artifacts_dir / "feature_columns.pkl")
    logger.info("Feature columns saved to artifacts/feature_columns.pkl")

    # SMOTE on train only — after encoding (needs numeric input)
    if cfg["smote"]["enabled"]:
        sm = SMOTE(random_state=cfg["smote"]["random_state"])
        X_train_enc, y_train = sm.fit_resample(X_train_enc, y_train)
        logger.info(
            f"SMOTE applied — resampled train size: {X_train_enc.shape[0]}"
        )

    return X_train_enc, X_test_enc, y_train


# ── 4. TRAIN ONE MODEL WITH HYPERPARAMETER SEARCH ────────────────────────────

def search_model(
    name: str,
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> object:
    """
    Run GridSearchCV or RandomizedSearchCV based on hyperparameters.yaml.
    Returns the best fitted estimator.
    """
    hp = load_hyperparameters()[name]

    common = dict(
        estimator=estimator,
        cv=hp["cv"],
        scoring=hp["scoring"],
        n_jobs=-1,
        verbose=1,
    )

    if hp["search"] == "grid":
        search = GridSearchCV(
            param_grid=hp["param_grid"],
            **common,
        )
    else:
        search = RandomizedSearchCV(
            param_distributions=hp["param_distributions"],
            n_iter=hp["n_iter"],
            random_state=42,
            **common,
        )

    search.fit(X_train, y_train)

    logger.info(f"{name} best params: {search.best_params_}")
    logger.info(f"{name} best CV score ({hp['scoring']}): {search.best_score_:.4f}")

    return search.best_estimator_


# ── 5. TRAIN ALL MODELS ───────────────────────────────────────────────────────

def train_all(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Train LR, RF, XGBoost — tune, evaluate, save each."""

    cfg          = load_config()
    artifacts_dir = PROJECT_ROOT / "artifacts" / "models"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest":       RandomForestClassifier(),
        "xgboost":             XGBClassifier(
                                   eval_metric="logloss",
                                   use_label_encoder=False,
                               ),
    }

    # human-readable names for plots / reports
    display_names = {
        "logistic_regression": "Logistic Regression",
        "random_forest":       "Random Forest",
        "xgboost":             "XGBoost",
    }

    # artifact save paths from config.yaml
    save_paths = {
        "logistic_regression": PROJECT_ROOT / cfg["paths"]["models"]["logistic"],
        "random_forest":       PROJECT_ROOT / cfg["paths"]["models"]["random_forest"],
        "xgboost":             PROJECT_ROOT / cfg["paths"]["models"]["xgboost"],
    }

    all_metrics  = []
    feature_names = joblib.load(PROJECT_ROOT / "artifacts" / "feature_columns.pkl")

    for key, estimator in models.items():
        display = display_names[key]
        logger.info(f"\n{'='*50}\nTraining {display}\n{'='*50}")

        best_model = search_model(key, estimator, X_train, y_train)

        # predict
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        # evaluate
        metrics, _ = evaluate_model(display, y_test, y_pred, y_prob)
        all_metrics.append(metrics)

        # plots
        plot_confusion_matrix(y_test, y_pred, display)
        plot_roc_curve(y_test, y_prob, display)
        plot_feature_importance(best_model, feature_names, display)

        # save model
        save_paths[key].parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, save_paths[key])
        logger.info(f"{display} saved to {save_paths[key]}")

    # comparison table + summary.md
    metrics_df = save_metrics(all_metrics)
    plot_model_comparison(metrics_df)
    save_summary(metrics_df)

    logger.info("\nAll models trained and evaluated.")


# ── 6. ENTRY POINT ───────────────────────────────────────────────────────────

def run_training() -> None:
    """Full training pipeline — called from main.py."""

    X_train, X_test, y_train, y_test = load_and_prepare()

    preprocessor = build_preprocessor()

    X_train_enc, X_test_enc, y_train_res = transform_and_resample(
        preprocessor, X_train, X_test, y_train
    )

    train_all(X_train_enc, X_test_enc, y_train_res, y_test)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
    )
    run_training()
