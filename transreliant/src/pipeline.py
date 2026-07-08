# src/pipeline.py

import numpy as np
import pandas as pd

from config_loader import load_config
from evaluate import load_stage1_artifacts, load_stage2_artifacts

NOT_CONFIRMED = 0
CONFIRMED = 1


def predict_cascade(raw_input_df: pd.DataFrame,
                     stage1_model, stage1_preprocessor,
                     stage2_model, stage2_preprocessor,
                     threshold: float,
                     cfg: dict) -> pd.DataFrame:

    numeric_cols = cfg["features"]["numerical"]
    categorical_cols = cfg["features"]["categorical"]
    feature_cols = numeric_cols + categorical_cols

    missing = set(feature_cols) - set(raw_input_df.columns)
    if missing:
        raise ValueError(f"raw_input_df is missing required feature columns: {missing}")

    X = raw_input_df[feature_cols]

    # ---- Stage 1: classify Confirmation Status ----
    X1_t = stage1_preprocessor.transform(X)
    proba_confirmed = stage1_model.predict_proba(X1_t)[:, 1]   # P(Confirmed)
    is_confirmed = proba_confirmed >= threshold
    is_not_confirmed = ~is_confirmed

    results = raw_input_df.copy()
    results["confirmation_prediction"] = np.where(is_confirmed, "Confirmed", "Not Confirmed")
    results["confirmation_confidence"] = np.where(is_confirmed, proba_confirmed, 1 - proba_confirmed)

    # ---- Stage 2: regress Waitlist Position, routed subset only ----
    results["estimated_waitlist_position"] = np.nan
    flagged = X[is_not_confirmed]

    if len(flagged) > 0:
        X2_t = stage2_preprocessor.transform(flagged)
        preds = stage2_model.predict(X2_t)
        results.loc[flagged.index, "estimated_waitlist_position"] = np.round(preds).astype(int)

    return results


def load_cascade_artifacts(cfg: dict):

    stage1_model, stage1_preprocessor = load_stage1_artifacts(cfg)
    stage2_model, stage2_preprocessor = load_stage2_artifacts(cfg)
    threshold = cfg.get("threshold", 0.5)
    return stage1_model, stage1_preprocessor, stage2_model, stage2_preprocessor, threshold


def main():
    """Smoke test: load real artifacts and run the cascade on a few
    held-out Stage 1 test rows, printing the result."""
    cfg = load_config()
    stage1_model, stage1_prep, stage2_model, stage2_prep, threshold = load_cascade_artifacts(cfg)

    from train_stage1 import load_stage1_splits
    _, X_test, _, _ = load_stage1_splits(cfg)
    sample = X_test.head(5)

    out = predict_cascade(sample, stage1_model, stage1_prep,
                           stage2_model, stage2_prep, threshold, cfg)
    print(out[["confirmation_prediction", "confirmation_confidence",
               "estimated_waitlist_position"]].to_string(index=False))


if __name__ == "__main__":
    main()