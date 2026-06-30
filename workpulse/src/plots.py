import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay

from src.config_loader import load_config


logger = logging.getLogger(__name__)

#plotting confusion matrix 
def plot_confusion_matrix(
    y_true,
    y_pred,
    model_name: str,
) -> None:
    """Plot confusion matrix."""

    cfg = load_config()

    figure_path = Path(
        cfg["paths"]["reports"]["figures"]
    )

    figure_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        cmap="Blues",
        colorbar=False,
    )

    plt.title(f"{model_name} Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        figure_path / f"{model_name}_confusion_matrix.png"
    )

    plt.close()

    logger.info(f"{model_name} confusion matrix saved.")

#plotting roc_curve
def plot_roc_curve(
    y_true,
    y_prob,
    model_name: str,
) -> None:
    """Plot ROC curve."""

    cfg = load_config()

    figure_path = Path(
        cfg["paths"]["reports"]["figures"]
    )

    figure_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(6, 5))

    RocCurveDisplay.from_predictions(
        y_true,
        y_prob,
    )

    plt.title(f"{model_name} ROC Curve")

    plt.tight_layout()

    plt.savefig(
        figure_path / f"{model_name}_roc_curve.png"
    )

    plt.close()

    logger.info(f"{model_name} ROC curve saved.")

#plotting feature_importance
def plot_feature_importance(
    model,
    feature_names,
    model_name: str,
    top_n: int = 10,
) -> None:
    """Plot feature importance."""

    cfg = load_config()

    figure_path = Path(
        cfg["paths"]["reports"]["figures"]
    )

    figure_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_

    elif hasattr(model, "coef_"):
        importance = abs(model.coef_[0])

    else:
        logger.warning(
            f"{model_name} does not support feature importance."
        )
        return

    importance_df = (
        pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": importance,
            }
        )
        .sort_values(
            by="Importance",
            ascending=False,
        )
        .head(top_n)
    )

    plt.figure(figsize=(8, 6))

    sns.barplot(
        data=importance_df,
        x="Importance",
        y="Feature",
    )

    plt.title(f"{model_name} Feature Importance")

    plt.tight_layout()

    plt.savefig(
        figure_path / f"{model_name}_feature_importance.png"
    )

    plt.close()

    logger.info(f"{model_name} feature importance saved.")

#plotting model comparision
def plot_model_comparison(
    metrics_df: pd.DataFrame,
) -> None:
    """Plot model comparison."""

    cfg = load_config()

    figure_path = Path(
        cfg["paths"]["reports"]["figures"]
    )

    figure_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(8, 5))

    comparison = metrics_df.set_index("Model")

    comparison.plot(
        kind="bar",
        figsize=(8, 5),
    )

    plt.ylabel("Score")

    plt.title("Model Comparison")

    plt.tight_layout()

    plt.savefig(
        figure_path / "model_comparison.png"
    )

    plt.close()

    logger.info("Model comparison saved.")

#main
if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
    )

    logger.info("plots.py loaded successfully.")