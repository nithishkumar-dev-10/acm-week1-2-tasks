import csv
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_score,recall_score,roc_curve,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from config_loader import load_config

try:
    from preprocessing import build_stage1_preprocessor
except ImportError:
    build_stage1_preprocessor = None

try:
    from utils import log_experiment
except ImportError:
    def log_experiment(stage=None, model_name=None, params=None,
                        cv_metric_name=None, cv_metric_mean=None, cv_metric_std=None,
                        test_metric_name=None, test_metric_value=None, **kwargs):
        
        log_path = Path("logs/experiment_log.csv")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not log_path.exists()
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(["timestamp", "stage", "model_name", "params",
                                  "cv_metric_name", "cv_metric_mean", "cv_metric_std",
                                  "test_metric_name", "test_metric_value"])
            writer.writerow([datetime.now().isoformat(), stage, model_name, params,
                              cv_metric_name, cv_metric_mean, cv_metric_std,
                              test_metric_name, test_metric_value])


NOT_CONFIRMED = 0
CONFIRMED = 1
PRECISION_FLOOR = 0.6
THRESHOLD_GRID = np.round(np.arange(0.10, 0.90 + 1e-9, 0.05), 2)


def load_stage1_test_split(cfg: dict):
    splits_dir = Path(cfg["data"]["splits_dir"])
    X_test = pd.read_csv(splits_dir / "stage1_X_test.csv")
    y_test = pd.read_csv(splits_dir / "stage1_y_test.csv").iloc[:, 0]
    return X_test, y_test


def load_stage1_train_split(cfg: dict):
    splits_dir = Path(cfg["data"]["splits_dir"])
    X_train = pd.read_csv(splits_dir / "stage1_X_train.csv")
    y_train = pd.read_csv(splits_dir / "stage1_y_train.csv").iloc[:, 0]
    return X_train, y_train


def _fallback_preprocessor(cfg: dict):
   
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    numeric_cols = cfg["features"]["numerical"]
    categorical_cols = cfg["features"]["categorical"]
    return ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])


def load_stage1_artifacts(cfg: dict):

    model_path = Path(cfg["artifacts"]["model_stage1"])
    prep_path = Path(cfg["artifacts"]["preprocessor_stage1"])

    if not model_path.exists():
        raise FileNotFoundError(
            f"Stage 1 model not found at {model_path.resolve()} — run train.py first."
        )
    model = joblib.load(model_path)
    print(f"Loaded Stage 1 model ({type(model).__name__}) from {model_path}")

    if prep_path.exists():
        preprocessor = joblib.load(prep_path)
        print(f"Loaded fitted preprocessor from {prep_path}")
    else:
        print(f"WARNING: {prep_path} not found. Rebuilding it by refitting on the "
              f"persisted train split ({cfg['data']['splits_dir']}stage1_X_train.csv) "
              f"using the Step 8 ColumnTransformer spec.")
        X_train, _ = load_stage1_train_split(cfg)
        preprocessor = (build_stage1_preprocessor(cfg) if build_stage1_preprocessor is not None
                         else _fallback_preprocessor(cfg))
        preprocessor.fit(X_train)
        prep_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, prep_path)
        print(f"Rebuilt and saved preprocessor -> {prep_path}")

    return model, preprocessor




def sweep_thresholds(y_true: pd.Series, y_proba: np.ndarray) -> pd.DataFrame:

    rows = []
    for t in THRESHOLD_GRID:
        y_pred = (y_proba >= t).astype(int)
        prec0 = precision_score(y_true, y_pred, pos_label=NOT_CONFIRMED, zero_division=0)
        rec0 = recall_score(y_true, y_pred, pos_label=NOT_CONFIRMED, zero_division=0)
        rows.append({"threshold": t, "precision_not_confirmed": prec0, "recall_not_confirmed": rec0})
    return pd.DataFrame(rows)


def pick_threshold(sweep_df: pd.DataFrame) -> float:
    
    MIN_USABLE_RECALL = 0.05  # below this, "meets the floor" is a near-empty-prediction artifact

    eligible = sweep_df[sweep_df["precision_not_confirmed"] >= PRECISION_FLOOR]
    usable = eligible[eligible["recall_not_confirmed"] >= MIN_USABLE_RECALL]

    if len(usable) > 0:
        best = usable.loc[usable["recall_not_confirmed"].idxmax()]
        print(f"Selected threshold {best['threshold']:.2f}: meets precision>=0.6 floor "
              f"(precision={best['precision_not_confirmed']:.4f}, "
              f"recall={best['recall_not_confirmed']:.4f})")
    elif len(eligible) > 0:
        degenerate = eligible.loc[eligible["recall_not_confirmed"].idxmax()]
        print("\n" + "!" * 78 +
              f"\nWARNING: threshold {degenerate['threshold']:.2f} technically clears "
              f"precision>=0.6 on the Not-Confirmed class,\nbut only because it predicts "
              f"'Not Confirmed' for almost no one (recall={degenerate['recall_not_confirmed']:.4f})."
              f"\nThat is a near-empty-prediction artifact of a model with ~no real signal "
              f"(see train.py docstring),\nnot a usable operating point. Falling back to the "
              f"best precision/recall trade-off across the full\ngrid instead — document this "
              f"finding ('the precision floor is only met in a degenerate, low-support "
              f"region') as a dataset limitation in the report.\n" + "!" * 78)
        usable_any = sweep_df[sweep_df["recall_not_confirmed"] >= MIN_USABLE_RECALL]
        best = usable_any.loc[usable_any["precision_not_confirmed"].idxmax()]
        print(f"Fallback threshold: {best['threshold']:.2f} "
              f"(precision={best['precision_not_confirmed']:.4f}, "
              f"recall={best['recall_not_confirmed']:.4f}) — "
              f"best precision among thresholds with non-degenerate recall.")
    else:
        usable_any = sweep_df[sweep_df["recall_not_confirmed"] >= MIN_USABLE_RECALL]
        pool = usable_any if len(usable_any) > 0 else sweep_df
        best = pool.loc[pool["precision_not_confirmed"].idxmax()]
        print("\n" + "!" * 78 +
              f"\nWARNING: no threshold in [0.10, 0.90] reaches precision>=0.6 on the "
              f"Not-Confirmed class.\nThis is consistent with the near-random-chance signal "
              f"documented in train.py, not a bug here.\nFalling back to the best available: "
              f"threshold={best['threshold']:.2f}, "
              f"precision={best['precision_not_confirmed']:.4f}, "
              f"recall={best['recall_not_confirmed']:.4f}\n" + "!" * 78)
    return float(best["threshold"])


def save_pr_curve(sweep_df: pd.DataFrame, chosen_threshold: float, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sweep_df["recall_not_confirmed"], sweep_df["precision_not_confirmed"],
             marker="o", label="threshold sweep")
    chosen_row = sweep_df.loc[sweep_df["threshold"] == chosen_threshold].iloc[0]
    ax.scatter([chosen_row["recall_not_confirmed"]], [chosen_row["precision_not_confirmed"]],
               color="red", zorder=5, s=80, label=f"chosen t={chosen_threshold:.2f}")
    ax.axhline(PRECISION_FLOOR, color="grey", linestyle="--", linewidth=1, label="precision floor (0.6)")
    ax.set_xlabel("Recall (Not Confirmed)")
    ax.set_ylabel("Precision (Not Confirmed)")
    ax.set_title("Stage 1 — Precision/Recall vs Threshold (Not Confirmed class)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved PR curve -> {out_path}")


def write_threshold_to_config(threshold: float, config_path: str = "config.yaml"):
    path = Path(config_path)
    with open(path, "r") as f:
        cfg_raw = yaml.safe_load(f)
    cfg_raw["threshold"] = round(float(threshold), 2)
    with open(path, "w") as f:
        yaml.safe_dump(cfg_raw, f, sort_keys=False)
    print(f"Wrote threshold={cfg_raw['threshold']} into {path}")



def save_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, out_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=[NOT_CONFIRMED, CONFIRMED])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Confirmed", "Confirmed"])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Stage 1 — Confusion Matrix (optimized threshold)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix -> {out_path}")
    return cm


def save_roc_curve(y_true: pd.Series, y_proba: np.ndarray, out_path: Path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance (AUC = 0.50)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Stage 1 — ROC Curve")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved ROC curve -> {out_path} (AUC={auc:.4f})")
    return auc


def save_feature_importance(model, preprocessor, out_path: Path, top_n: int = 20):
    if not hasattr(model, "feature_importances_"):
        print(f"Model {type(model).__name__} has no .feature_importances_ — "
              f"skipping feature importance plot (use permutation importance instead if needed).")
        return None

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"f{i}" for i in range(len(model.feature_importances_))]

    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(importances))))
    importances.iloc[::-1].plot.barh(ax=ax)
    ax.set_xlabel("Importance")
    ax.set_title(f"Stage 1 — Feature Importance ({type(model).__name__}, top {len(importances)})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved feature importance -> {out_path}")
    return importances


def save_metrics_csv(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray,
                      threshold: float, out_path: Path):
    report = classification_report(
        y_true, y_pred, target_names=["Not Confirmed", "Confirmed"], output_dict=True
    )
    auc = roc_auc_score(y_true, y_proba)

    rows = []
    for label, stats in report.items():
        if label == "accuracy":
            rows.append({"class": "accuracy", "precision": None, "recall": None,
                         "f1_score": None, "support": None, "value": stats})
        else:
            rows.append({"class": label, "precision": stats["precision"], "recall": stats["recall"],
                         "f1_score": stats["f1-score"], "support": stats["support"], "value": None})
    rows.append({"class": "ROC_AUC", "precision": None, "recall": None,
                 "f1_score": None, "support": None, "value": auc})
    rows.append({"class": "threshold_used", "precision": None, "recall": None,
                 "f1_score": None, "support": None, "value": threshold})

    metrics_df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(out_path, index=False)
    print(f"Saved Stage 1 metrics -> {out_path}")
    return metrics_df



def evaluate_stage1(cfg: dict):
    model, preprocessor = load_stage1_artifacts(cfg)
    X_test, y_test = load_stage1_test_split(cfg)

    X_test_t = preprocessor.transform(X_test)
    y_proba = model.predict_proba(X_test_t)[:, 1]  # P(Confirmed)

    # ---- Step 12: threshold optimization ----
    sweep_df = sweep_thresholds(y_test, y_proba)
    print("\nThreshold sweep (Not Confirmed class):")
    print(sweep_df.to_string(index=False))

    chosen_threshold = pick_threshold(sweep_df)
    save_pr_curve(sweep_df, chosen_threshold, Path("reports/figures/pr_curve_stage1.png"))
    write_threshold_to_config(chosen_threshold, config_path="config.yaml")

    log_experiment(
        stage="1", model_name=f"{type(model).__name__}_threshold_selection",
        test_metric_name="chosen_threshold", test_metric_value=chosen_threshold,
    )

    # ---- Step 13: Stage 1 evaluation using the optimized threshold ----
    y_pred = (y_proba >= chosen_threshold).astype(int)

    save_confusion_matrix(y_test, y_pred, Path("reports/figures/confusion_matrix_stage1.png"))
    auc = save_roc_curve(y_test, y_proba, Path("reports/figures/roc_stage1.png"))
    save_feature_importance(model, preprocessor, Path("reports/figures/feature_importance_stage1.png"))
    save_metrics_csv(y_test, y_pred, y_proba, chosen_threshold, Path("reports/metrics/stage1_metrics.csv"))

    log_experiment(
        stage="1", model_name=f"{type(model).__name__}_evaluate_TEST",
        test_metric_name="roc_auc", test_metric_value=auc,
    )

    print(f"\nSteps 12+13 complete. Threshold={chosen_threshold:.2f} written to config.yaml. "
          f"Figures saved in reports/figures/, metrics in reports/metrics/stage1_metrics.csv.")

    return chosen_threshold


def load_stage2_test_split(cfg: dict):
    splits_dir = Path(cfg["data"]["splits_dir"])
    X_test = pd.read_csv(splits_dir / "stage2_X_test.csv")
    y_test = pd.read_csv(splits_dir / "stage2_y_test.csv").iloc[:, 0]
    return X_test, y_test


def load_stage2_artifacts(cfg: dict):
    model_path = Path(cfg["artifacts"]["model_stage2"])
    prep_path = Path(cfg["artifacts"]["preprocessor_stage2"])

    if not model_path.exists():
        raise FileNotFoundError(
            f"Stage 2 model not found at {model_path.resolve()} — run train_stage2.py first."
        )
    if not prep_path.exists():
        raise FileNotFoundError(
            f"Stage 2 preprocessor not found at {prep_path.resolve()} — run train_stage2.py first."
        )

    model = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)
    print(f"Loaded Stage 2 model ({type(model).__name__}) from {model_path}")
    print(f"Loaded Stage 2 preprocessor from {prep_path}")
    return model, preprocessor


def save_pred_vs_actual(y_true: pd.Series, y_pred: np.ndarray, out_path: Path):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(y_true, y_pred, alpha=0.3, s=12)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="red", label="Identity (perfect prediction)")
    ax.set_xlabel("Actual Waitlist Position")
    ax.set_ylabel("Predicted Waitlist Position")
    ax.set_title("Stage 2 — Predicted vs Actual")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved predicted-vs-actual plot -> {out_path}")


def save_residuals_plot(y_true: pd.Series, y_pred: np.ndarray, out_path: Path):
    residuals = np.asarray(y_true) - y_pred
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, alpha=0.3, s=12)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Predicted Waitlist Position")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title("Stage 2 — Residuals vs Predicted")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved residuals plot -> {out_path}")


def save_stage2_metrics_csv(rmse: float, mae: float, r2: float, out_path: Path):
    metrics_df = pd.DataFrame([
        {"metric": "RMSE", "value": rmse},
        {"metric": "MAE", "value": mae},
        {"metric": "R2", "value": r2},
    ])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(out_path, index=False)
    print(f"Saved Stage 2 metrics -> {out_path}")
    return metrics_df


def evaluate_stage2(cfg: dict):
    model, preprocessor = load_stage2_artifacts(cfg)
    X_test, y_test = load_stage2_test_split(cfg)

    X_test_t = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_t)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    print(f"\n=== Stage 2 held-out test performance ({type(model).__name__}) ===")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE:  {mae:.4f}")
    print(f"Test R2:   {r2:.4f}")

    save_pred_vs_actual(y_test, y_pred, Path("reports/figures/pred_vs_actual_stage2.png"))
    save_residuals_plot(y_test, y_pred, Path("reports/figures/residuals_stage2.png"))
    save_stage2_metrics_csv(rmse, mae, r2, Path("reports/metrics/stage2_metrics.csv"))

    log_experiment(
        stage="2", model_name=f"{type(model).__name__}_evaluate_TEST",
        test_metric_name="rmse", test_metric_value=rmse,
    )
    log_experiment(
        stage="2", model_name=f"{type(model).__name__}_evaluate_TEST",
        test_metric_name="r2", test_metric_value=r2,
    )

    print(f"\nStep 16 complete. Figures saved in reports/figures/, "
          f"metrics in reports/metrics/stage2_metrics.csv.")




def _load_stage2_own_test_rmse(cfg: dict):
    
    path = Path("reports/metrics/stage2_metrics.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path)
    row = df[df["metric"] == "RMSE"]
    if row.empty:
        return None
    return float(row["value"].iloc[0])


def evaluate_system(cfg: dict):
    
    from train_stage1 import load_featured_data, split_and_save_stage1
    from pipeline import load_cascade_artifacts, predict_cascade

    df = load_featured_data(cfg)
    _, X1_test, _, y1_test = split_and_save_stage1(df, cfg)

    stage1_model, stage1_prep, stage2_model, stage2_prep, threshold = load_cascade_artifacts(cfg)

    # ---- Run the full cascade over the entire Stage 1 test set ----
    cascade_out = predict_cascade(
        X1_test, stage1_model, stage1_prep, stage2_model, stage2_prep, threshold, cfg
    )

    y_pred = (cascade_out["confirmation_prediction"] == "Confirmed").astype(int)
    y_pred.index = X1_test.index
    # Reconstruct P(Confirmed) from the stored confidence (which is
    # P(predicted class), not always P(Confirmed) directly).
    proba_confirmed = np.where(
        y_pred.values == 1,
        cascade_out["confirmation_confidence"].values,
        1 - cascade_out["confirmation_confidence"].values,
    )

    # ---- End-to-end classification metrics (system-level restatement of Step 13) ----
    report = classification_report(
        y1_test, y_pred, target_names=["Not Confirmed", "Confirmed"], output_dict=True, zero_division=0
    )
    system_auc = roc_auc_score(y1_test, proba_confirmed)

    # ---- Coverage: % of test set where Stage 2 even fires ----
    coverage = float((y_pred == 0).mean())

    # ---- Stage 2 RMSE/MAE on the subset Stage 1 actually routed to it ----
    # (actually-Not-Confirmed AND correctly flagged by Stage 1 — NOT the
    # same population as Step 16's clean Stage-2-only test split.)
    status_col = cfg["target"]["stage1"]
    waitlist_col = cfg["target"]["stage2"]
    actually_not_confirmed = (y1_test == 0)
    correctly_routed = actually_not_confirmed & (y_pred == 0)
    routed_idx = y1_test.index[correctly_routed]

    if len(routed_idx) > 0:
        actual_waitlist = df.loc[routed_idx, waitlist_col]
        predicted_waitlist = cascade_out.loc[routed_idx, "estimated_waitlist_position"]
        routed_rmse = float(np.sqrt(mean_squared_error(actual_waitlist, predicted_waitlist)))
        routed_mae = float(mean_absolute_error(actual_waitlist, predicted_waitlist))
    else:
        routed_rmse, routed_mae = float("nan"), float("nan")
        print("WARNING: Stage 1 correctly routed zero rows to Stage 2 in this test set — "
              "routed-subset RMSE/MAE are undefined (NaN).")

    own_test_rmse = _load_stage2_own_test_rmse(cfg)

    print(f"\n=== Step 20: system-level (cascade) evaluation ===")
    print(f"System AUC (Confirmed vs Not Confirmed): {system_auc:.4f}")
    print(f"Coverage (% test set routed to Stage 2): {coverage:.4f}")
    print(f"Stage 2 RMSE on Stage-1-routed subset: {routed_rmse:.4f}  (n={len(routed_idx)})")
    if own_test_rmse is not None:
        print(f"Stage 2 RMSE on its own clean test split (Step 16, for comparison): {own_test_rmse:.4f}")
        print("Note: these two RMSE numbers differ because the routed-subset population "
              "includes rows Stage 1 mis-flagged, while Step 16's split is ground-truth-clean.")

    # ---- Save reports/metrics/system_metrics.csv ----
    rows = []
    for label, stats in report.items():
        if label == "accuracy":
            rows.append({"metric": "accuracy", "value": stats})
        else:
            rows.append({"metric": f"{label}_precision", "value": stats["precision"]})
            rows.append({"metric": f"{label}_recall", "value": stats["recall"]})
            rows.append({"metric": f"{label}_f1", "value": stats["f1-score"]})
    rows.append({"metric": "system_roc_auc", "value": system_auc})
    rows.append({"metric": "coverage_pct_routed_to_stage2", "value": coverage})
    rows.append({"metric": "stage2_rmse_on_routed_subset", "value": routed_rmse})
    rows.append({"metric": "stage2_mae_on_routed_subset", "value": routed_mae})
    rows.append({"metric": "stage2_rmse_own_clean_test_split_step16", "value": own_test_rmse})
    rows.append({"metric": "threshold_used", "value": threshold})
    rows.append({"metric": "n_routed_to_stage2", "value": len(routed_idx)})

    out_path = Path("reports/metrics/system_metrics.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved system-level metrics -> {out_path}")

    log_experiment(
        stage="system", model_name="cascade",
        test_metric_name="system_roc_auc", test_metric_value=system_auc,
    )
    log_experiment(
        stage="system", model_name="cascade",
        test_metric_name="coverage_pct_routed_to_stage2", test_metric_value=coverage,
    )
    log_experiment(
        stage="system", model_name="cascade",
        test_metric_name="stage2_rmse_on_routed_subset", test_metric_value=routed_rmse,
    )

    print("Step 20 complete.\n")
    return {
        "system_auc": system_auc,
        "coverage": coverage,
        "routed_rmse": routed_rmse,
        "routed_mae": routed_mae,
        "own_test_rmse": own_test_rmse,
    }




def main():
    cfg = load_config()
    evaluate_stage1(cfg)
    evaluate_stage2(cfg)
    evaluate_system(cfg)


if __name__ == "__main__":
    main()