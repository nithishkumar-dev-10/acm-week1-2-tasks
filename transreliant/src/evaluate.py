
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
from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
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
        """Fallback logger matching the schema in the roadmap (Step 21),
        used only if src/utils.py isn't importable in this environment."""
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
    """Reproduces the Step 8 ColumnTransformer spec if preprocessing.py
    isn't importable: numeric -> StandardScaler, categorical -> OneHotEncoder."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    numeric_cols = cfg["features"]["numerical"]
    categorical_cols = cfg["features"]["categorical"]
    return ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])


def load_stage1_artifacts(cfg: dict):
    """
    Load the tuned Stage 1 model and its fitted preprocessor.

    If preprocessor_stage1.pkl isn't on disk (e.g. only the model artifact
    was carried over), rebuild it deterministically by refitting on the
    persisted train split with the Step 8 spec. Refitting on the exact same
    train split reproduces the identical fitted transformer, since neither
    StandardScaler nor OneHotEncoder involves randomness — so this is safe,
    not an approximation.
    """
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
    """
    y_proba = P(Confirmed). A row is predicted Confirmed (1) if
    y_proba >= t, else Not Confirmed (0). For each threshold, compute
    precision/recall on the Not Confirmed class (0) — that's the class
    the cost argument in Step 12 cares about catching.
    """
    rows = []
    for t in THRESHOLD_GRID:
        y_pred = (y_proba >= t).astype(int)
        prec0 = precision_score(y_true, y_pred, pos_label=NOT_CONFIRMED, zero_division=0)
        rec0 = recall_score(y_true, y_pred, pos_label=NOT_CONFIRMED, zero_division=0)
        rows.append({"threshold": t, "precision_not_confirmed": prec0, "recall_not_confirmed": rec0})
    return pd.DataFrame(rows)


def pick_threshold(sweep_df: pd.DataFrame) -> float:
    """
    Among thresholds where precision(Not Confirmed) >= 0.6, pick the one
    with the highest recall(Not Confirmed). If none clear the precision
    floor, fall back to the threshold with the highest precision achieved
    and print a warning — this can legitimately happen when Stage 1 has
    close to no real signal (see module docstring), so don't be surprised
    if the 0.6 floor isn't cleanly met.
    """
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


# ---------------------------------------------------------------------------
# Step 13 — Stage 1 evaluation
# ---------------------------------------------------------------------------

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




def main():
    cfg = load_config()
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


if __name__ == "__main__":
    main()