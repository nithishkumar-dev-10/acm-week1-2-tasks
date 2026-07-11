
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

from config_loader import load_config, get_path
from preprocessing import build_stage2_preprocessor
from utils import log_experiment

AAA_LOW_R2_THRESHOLD = 0.05  # below this, we flag rather than declare victory



def load_featured_data(cfg: dict) -> pd.DataFrame:
    path = Path(get_path(cfg, "data", "featured"))
    if not path.exists():
        raise FileNotFoundError(f"Featured data not found at {path.resolve()} — run feature_engineering.py first.")
    df = pd.read_csv(path)
    print(f"Loaded featured data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def build_stage2_subset(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    
    status_col = cfg["target"]["stage1"]  # "Confirmation Status"
    stage2_df = df[df[status_col] == 0].copy()
    print(f"Stage 2 subset: {stage2_df.shape[0]} genuinely Not-Confirmed rows "
          f"(out of {df.shape[0]} total).")
    return stage2_df


def split_and_save_stage2(stage2_df: pd.DataFrame, cfg: dict):
    
    target_col = cfg["target"]["stage2"]  # "Waitlist Position"
    numeric_cols = cfg["features"]["numerical"]
    categorical_cols = cfg["features"]["categorical"]
    feature_cols = numeric_cols + categorical_cols  # never includes Confirmation Status

    X = stage2_df[feature_cols]
    y = stage2_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=cfg["random_seed"]
    )

    splits_dir = Path(cfg["data"]["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(splits_dir / "stage2_X_train.csv", index=False)
    X_test.to_csv(splits_dir / "stage2_X_test.csv", index=False)
    y_train.to_csv(splits_dir / "stage2_y_train.csv", index=False)
    y_test.to_csv(splits_dir / "stage2_y_test.csv", index=False)

    print(f"Stage 2 split saved: train={X_train.shape[0]} rows, test={X_test.shape[0]} rows "
          f"(target range: {y.min():.0f}-{y.max():.0f}, mean={y.mean():.1f})")
    return X_train, X_test, y_train, y_test


def load_stage2_splits(cfg: dict):
    """Re-load the persisted Stage 2 splits (used when resuming without re-deriving them)."""
    splits_dir = Path(cfg["data"]["splits_dir"])
    X_train = pd.read_csv(splits_dir / "stage2_X_train.csv")
    X_test = pd.read_csv(splits_dir / "stage2_X_test.csv")
    y_train = pd.read_csv(splits_dir / "stage2_y_train.csv").iloc[:, 0]
    y_test = pd.read_csv(splits_dir / "stage2_y_test.csv").iloc[:, 0]
    print(f"Loaded Stage 2 splits: train={X_train.shape[0]}, test={X_test.shape[0]}")
    return X_train, X_test, y_train, y_test




def compare_models(X_train, y_train, cfg: dict) -> dict:
    
    models = {
        "ridge": Ridge(random_state=cfg["random_seed"]),
        "rf": RandomForestRegressor(random_state=cfg["random_seed"]),
        "xgb": XGBRegressor(random_state=cfg["random_seed"]),
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=cfg["random_seed"])
    results = {}

    for name, model in models.items():
        preprocessor = build_stage2_preprocessor(cfg)
        pipe = Pipeline([("prep", preprocessor), ("model", model)])

        neg_rmse_scores = cross_val_score(
            pipe, X_train, y_train, cv=kf, scoring="neg_root_mean_squared_error"
        )
        rmse_scores = -neg_rmse_scores  # flip back to positive RMSE for readability

        log_experiment(
            stage="2", model_name=name,
            cv_metric_name="rmse", cv_metric_mean=rmse_scores.mean(), cv_metric_std=rmse_scores.std(),
        )

        results[name] = {"rmse_mean": rmse_scores.mean(), "rmse_std": rmse_scores.std()}
        print(f"{name}: RMSE={rmse_scores.mean():.4f} (+/-{rmse_scores.std():.4f})")

    return results


def pick_best_baseline_model(results: dict) -> str:
    best_name = min(results, key=lambda name: results[name]["rmse_mean"])
    print(f"Best model by mean CV RMSE: {best_name} (RMSE={results[best_name]['rmse_mean']:.4f})")
    return best_name



def get_search_space(model_name: str, cfg: dict) -> dict:
    
    spaces = {
        "ridge": {
            "estimator": Ridge(random_state=cfg["random_seed"]),
            "params": {
                "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
            },
        },
        "rf": {
            "estimator": RandomForestRegressor(random_state=cfg["random_seed"]),
            "params": {
                "model__n_estimators": [200, 400, 600],
                "model__max_depth": [3, 5, 7, None],
                "model__min_samples_leaf": [1, 5, 20],
            },
        },
        "xgb": {
            "estimator": XGBRegressor(random_state=cfg["random_seed"]),
            "params": {
                "model__max_depth": [3, 5, 7],
                "model__n_estimators": [200, 400, 600],
                "model__learning_rate": [0.01, 0.05, 0.1],
            },
        },
    }
    return spaces[model_name]


def tune_winner(winner_name: str, X_train, y_train, cfg: dict):
    kf = KFold(n_splits=5, shuffle=True, random_state=cfg["random_seed"])
    spec = get_search_space(winner_name, cfg)

    preprocessor = build_stage2_preprocessor(cfg)
    pipe = Pipeline([("prep", preprocessor), ("model", spec["estimator"])])

    search = GridSearchCV(
        pipe,
        param_grid=spec["params"],
        scoring="neg_root_mean_squared_error",
        cv=kf,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    best_pipe = search.best_estimator_
    cv_rmse_mean = -search.best_score_
    cv_rmse_std = search.cv_results_["std_test_score"][search.best_index_]

    log_experiment(
        stage="2", model_name=f"{winner_name}_tuned",
        params=search.best_params_,
        cv_metric_name="rmse", cv_metric_mean=cv_rmse_mean, cv_metric_std=cv_rmse_std,
    )

    print(f"\n{winner_name}: best CV RMSE={cv_rmse_mean:.4f} (+/-{cv_rmse_std:.4f})  "
          f"params={search.best_params_}")

    return best_pipe, cv_rmse_mean, cv_rmse_std, search.best_params_



def evaluate_on_test(best_pipe, X_test, y_test, model_name: str):
    y_pred = best_pipe.predict(X_test)

    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    print(f"\n=== Held-out test performance ({model_name}) ===")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test MAE:  {test_mae:.4f}")
    print(f"Test R2:   {test_r2:.4f}")

    log_experiment(
        stage="2", model_name=f"{model_name}_tuned_TEST",
        test_metric_name="rmse", test_metric_value=test_rmse,
    )
    log_experiment(
        stage="2", model_name=f"{model_name}_tuned_TEST",
        test_metric_name="mae", test_metric_value=test_mae,
    )
    log_experiment(
        stage="2", model_name=f"{model_name}_tuned_TEST",
        test_metric_name="r2", test_metric_value=test_r2,
    )

    if test_r2 < AAA_LOW_R2_THRESHOLD:
        print(
            "\n" + "!" * 78 +
            f"\nDIAGNOSTIC: test R2 ({test_r2:.4f}) is at/near zero — the model explains "
            "almost none\nof the variance in Waitlist Position. This lines up with the EDA "
            "observation that\nWaitlist Position looks close to uniformly distributed across "
            "1-200 (see roadmap\nStep 5 / Limitations note) rather than driven by booking/"
            "journey features the way\na real waitlist queue would be. Treat this as a "
            "DATASET finding, not a modeling\nfailure — document it in the report's "
            "Limitations section rather than sinking more\ntime into re-tuning Stage 2.\n"
            + "!" * 78
        )

    return test_rmse, test_mae, test_r2


def save_artifacts(best_pipe, cfg: dict):
    prep_path = Path(cfg["artifacts"]["preprocessor_stage2"])
    model_path = Path(cfg["artifacts"]["model_stage2"])
    prep_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_pipe.named_steps["prep"], prep_path)
    joblib.dump(best_pipe.named_steps["model"], model_path)
    print(f"\nSaved Stage 2 preprocessor -> {prep_path}")
    print(f"Saved Stage 2 model        -> {model_path}")




def main():
    cfg = load_config()

    # Step 14: build the ground-truth Not-Confirmed subset, split, persist
    df = load_featured_data(cfg)
    stage2_df = build_stage2_subset(df, cfg)
    X_train, X_test, y_train, y_test = split_and_save_stage2(stage2_df, cfg)

    # Step 15: compare Ridge / RF / XGB, tune the winner
    baseline_results = compare_models(X_train, y_train, cfg)
    winner_name = pick_best_baseline_model(baseline_results)

    best_pipe, cv_rmse_mean, cv_rmse_std, best_params = tune_winner(winner_name, X_train, y_train, cfg)

    evaluate_on_test(best_pipe, X_test, y_test, winner_name)
    save_artifacts(best_pipe, cfg)

    print(f"\nStage 2 training complete. Tuned '{winner_name}' selected by CV RMSE and saved as Stage 2 artifact.")


if __name__ == "__main__":
    main()