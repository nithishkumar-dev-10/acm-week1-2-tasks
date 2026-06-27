import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from src.config_loader import PROJECT_ROOT, load_config

logger = logging.getLogger(__name__)


def evaluate_model(
    model_name: str,
    y_true,
    y_pred,
    y_prob,
) -> dict:
    """Compute all required metrics for one model and return as dict."""

    accuracy  = accuracy_score(y_true, y_pred)
    f1_macro  = f1_score(y_true, y_pred, average="macro")
    roc_auc   = roc_auc_score(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred)

    # cm layout for binary: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0   # predicted hired but not hired
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0   # predicted not hired but was hired

    report = classification_report(y_true, y_pred, output_dict=True)

    metrics = {
        "Model":    model_name,
        "Accuracy": round(accuracy, 4),
        "F1-macro": round(f1_macro, 4),
        "ROC-AUC":  round(roc_auc, 4),
        "FPR":      round(fpr, 4),
        "FNR":      round(fnr, 4),
    }

    logger.info(
        f"{model_name} — Accuracy: {accuracy:.4f} | "
        f"F1-macro: {f1_macro:.4f} | ROC-AUC: {roc_auc:.4f} | "
        f"FPR: {fpr:.4f} | FNR: {fnr:.4f}"
    )

    return metrics, report


def save_metrics(all_metrics: list[dict]) -> pd.DataFrame:
    """Save all model metrics to reports/metrics/metrics.csv and return DataFrame."""

    cfg = load_config()

    metrics_path = PROJECT_ROOT / cfg["paths"]["reports"]["metrics"]
    metrics_path.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(metrics_path / "metrics.csv", index=False)

    logger.info(f"Metrics saved to {metrics_path / 'metrics.csv'}")

    return metrics_df


def save_summary(metrics_df: pd.DataFrame) -> None:
    """Write reports/summary.md with the final model comparison table."""

    cfg = load_config()

    summary_path = PROJECT_ROOT / cfg["paths"]["reports"]["summary"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # pick best model by F1-macro, break ties by ROC-AUC
    best_idx    = metrics_df.sort_values(
        ["F1-macro", "ROC-AUC"], ascending=False
    ).index[0]
    best_model  = metrics_df.loc[best_idx, "Model"]

    lines = [
        "# TalentIQ — Model Comparison\n\n",
        "## Results\n\n",
        "| Model | Accuracy | F1-macro | ROC-AUC | FPR | FNR | Verdict |\n",
        "|---|---|---|---|---|---|---|\n",
    ]

    verdicts = {"Logistic Regression": "Baseline", "Random Forest": "Compare", "XGBoost": "Compare"}
    verdicts[best_model] = "✅ Winner"

    for _, row in metrics_df.iterrows():
        lines.append(
            f"| {row['Model']} | {row['Accuracy']} | {row['F1-macro']} | "
            f"{row['ROC-AUC']} | {row['FPR']} | {row['FNR']} | {verdicts.get(row['Model'], '')} |\n"
        )

    lines.append(
        f"\n**Selected model: {best_model}** — highest F1-macro.\n"
    )

    with open(summary_path, "w") as f:
        f.writelines(lines)

    logger.info(f"Summary saved to {summary_path}")
