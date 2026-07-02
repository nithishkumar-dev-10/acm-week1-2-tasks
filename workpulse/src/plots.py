import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

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

#plotting shap summary (explainability)
def plot_shap_summary(
    model,
    X,
    feature_names,
    model_name: str,
    max_display: int = 10,
):
    """
    Compute SHAP values for the given model and save a beeswarm summary plot.
    Returns the fitted explainer (so it can be pickled/reused at inference time),
    or None if SHAP failed for this model type.
    """

    cfg = load_config()

    figure_path = Path(
        cfg["paths"]["reports"]["figures"]
    )

    figure_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=feature_names)

    try:
        if hasattr(model, "feature_importances_"):
            # Tree-based models: RandomForest, XGBoost
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_df)

            # Binary-classification TreeExplainer sometimes returns a list [class0, class1]
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            # ...or a single 3D array shaped (n_samples, n_features, n_classes)
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]

        elif hasattr(model, "coef_"):
            # Linear models: LogisticRegression
            explainer = shap.LinearExplainer(model, X_df)
            shap_values = explainer.shap_values(X_df)

        else:
            logger.warning(
                f"{model_name} does not support SHAP explanation for this model type."
            )
            return None

    except Exception as e:
        logger.warning(f"{model_name}: SHAP explainer failed ({e}), skipping SHAP plot.")
        return None

    plt.figure(figsize=(8, 6))

    shap.summary_plot(
        shap_values,
        X_df,
        max_display=max_display,
        show=False,
    )

    plt.title(f"{model_name} SHAP Summary")

    plt.tight_layout()

    plt.savefig(
        figure_path / f"{model_name}_shap_summary.png"
    )

    plt.close()

    logger.info(f"{model_name} SHAP summary plot saved.")

    return explainer

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

#plotting class_distribution
def plot_class_distribution(
    y,
    target_name: str = "Target",
) -> None:
    """Plot class distribution of the target variable."""

    cfg = load_config()

    figure_path = Path(
        cfg["paths"]["reports"]["figures"]
    )

    figure_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    class_counts = (
        pd.Series(y)
        .value_counts()
        .sort_index()
    )

    plt.figure(figsize=(6, 5))

    sns.barplot(
        x=class_counts.index.astype(str),
        y=class_counts.values,
    )

    plt.xlabel(target_name)

    plt.ylabel("Count")

    plt.title(f"{target_name} Class Distribution")

    plt.tight_layout()

    plt.savefig(
        figure_path / "class_distribution.png"
    )

    plt.close()

    logger.info("Class distribution plot saved.")

#plotting boxplots
def plot_boxplots(
    df: pd.DataFrame,
    numeric_cols,
) -> None:
    """Plot boxplots for numeric features to inspect spread and outliers."""

    cfg = load_config()

    figure_path = Path(
        cfg["paths"]["reports"]["figures"]
    )

    figure_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    n_cols = 3

    n_rows = -(-len(numeric_cols) // n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
    )

    axes = axes.flatten()

    for idx, col in enumerate(numeric_cols):

        sns.boxplot(
            data=df,
            y=col,
            ax=axes[idx],
        )

        axes[idx].set_title(col)

    for idx in range(len(numeric_cols), len(axes)):

        fig.delaxes(axes[idx])

    fig.suptitle("Feature Boxplots")

    plt.tight_layout()

    plt.savefig(
        figure_path / "boxplots.png"
    )

    plt.close()

    logger.info("Boxplots saved.")

#main
if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
    )

    logger.info("plots.py loaded successfully.")