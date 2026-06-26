"""
src/plots.py
TalentIQ — Phase 4
All plots for evaluation. No pipeline.
- RF    → feature_importances_
- LR    → coef_ magnitude
- XGB   → SHAP TreeExplainer
- All   → ROC curves, Confusion matrices
Saved to reports/figures/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

FIGURES_DIR = "reports/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# 1. RANDOM FOREST — Feature Importances

def plot_rf_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1]
    names       = [feature_names[i] for i in indices]
    values      = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(names[::-1], values[::-1], color="#2196F3")
    ax.set_xlabel("Importance Score")
    ax.set_title("Random Forest — Feature Importances")
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "rf_feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved → {path}")

# LOGISTIC REGRESSION — Coefficient Magnitude 

def plot_lr_coefficients(model, feature_names):
    coefs   = np.abs(model.coef_[0])
    indices = np.argsort(coefs)[::-1]
    names   = [feature_names[i] for i in indices]
    values  = coefs[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(names[::-1], values[::-1], color="#4CAF50")
    ax.set_xlabel("|Coefficient|")
    ax.set_title("Logistic Regression — Coefficient Magnitudes")
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "lr_coefficients.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved → {path}")

#  XGBOOST — SHAP Values 

def plot_xgb_shap(model, X_test):
    try:
        import shap
    except ImportError:
        print("[WARN] shap not installed → pip install shap")
        return

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Bar summary
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title("XGBoost — SHAP Feature Importance (Bar)")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "xgb_shap_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved → {path}")

    # Dot summary (shows direction)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("XGBoost — SHAP Summary (Direction)")
    plt.tight_layout()
    path2 = os.path.join(FIGURES_DIR, "xgb_shap_dot.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved → {path2}")

# ROC CURVES — All 3 Models 

def plot_roc_curves(models_dict, X_test, y_test):
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=(8, 6))
    colors  = {"Logistic Regression": "#2196F3", "Random Forest": "#4CAF50", "XGBoost": "#FF5722"}

    for name, model in models_dict.items():
        y_prob       = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _  = roc_curve(y_test, y_prob)
        roc_auc      = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", color=colors[name], lw=2)

    ax.plot([0,1], [0,1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All 3 Models")
    ax.legend(loc="lower right")
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "roc_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved → {path}")

# CONFUSION MATRICES — All 3 Models

def plot_confusion_matrices(models_dict, X_test, y_test):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, (name, model) in zip(axes, models_dict.items()):
        cm   = confusion_matrix(y_test, model.predict(X_test))
        disp = ConfusionMatrixDisplay(cm, display_labels=["Not Hired", "Hired"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name)

    plt.suptitle("Confusion Matrices — All 3 Models", fontsize=13, y=1.02)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved → {path}")

# MAIN

def run_plots():
    print(f"\n{'='*50}")
    print("  GENERATING PLOTS")
    print(f"{'='*50}")

    lr_model  = joblib.load("artifacts/models/logistic_regression.pkl")
    rf_model  = joblib.load("artifacts/models/random_forest.pkl")
    xgb_model = joblib.load("artifacts/models/xgboost.pkl")

    test_df      = pd.read_csv("data/splits/test.csv")
    feature_cols = joblib.load("artifacts/feature_columns.pkl")
    X_test       = test_df[feature_cols]
    y_test       = test_df["Hired"]

    models_dict  = {
        "Logistic Regression": lr_model,
        "Random Forest":       rf_model,
        "XGBoost":             xgb_model,
    }
    feature_names = list(X_test.columns)

    plot_rf_feature_importance(rf_model,  feature_names)
    plot_lr_coefficients(lr_model,        feature_names)
    plot_xgb_shap(xgb_model,              X_test)
    plot_roc_curves(models_dict,           X_test, y_test)
    plot_confusion_matrices(models_dict,   X_test, y_test)

    print(f"\n[DONE] All plots saved to {FIGURES_DIR}/\n")

if __name__ == "__main__":
    run_plots()
