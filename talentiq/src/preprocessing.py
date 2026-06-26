"""
src/preprocessing.py
TalentIQ — Phase 2
Steps: Load → Duplicates → Impute → Outliers → Split → Encode → Scale → Save

CHANGE FROM v1: Encoding moved to AFTER the train/test split, and encoders
are now real sklearn objects (OrdinalEncoder, OneHotEncoder) fit on TRAIN
ONLY and persisted to artifacts/encoder.pkl — same train/serve symmetry
principle already applied to the scaler. The old version used df.map() and
pd.get_dummies() directly on the full dataframe, which doesn't produce a
reusable encoder object. That meant a new candidate at inference time had
no guaranteed way to be encoded into the same columns the model was trained
on (e.g. get_dummies silently drops/shifts columns if a category is missing
from a single-row input). This version fixes that.

Mode-aware: sample (testing) / full (training) via config.yaml
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from src.config_loader import load_config, load_features, load_data

# ── 1. DROP DUPLICATES ────────────────────────────────────────────────────────

def drop_duplicates(df):
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[INFO] Dropped {before - len(df)} duplicate rows → {len(df)} remaining")
    return df

# ── 2. IMPUTE MISSING VALUES ──────────────────────────────────────────────────

def impute_missing(df, feat_cfg):
    for col in feat_cfg["numerical_features"]:
        if col in df.columns and df[col].isnull().sum() > 0:
            val = df[col].median()
            df[col] = df[col].fillna(val)
            print(f"[INFO] Imputed '{col}' → median={val:.2f}")

    for col in feat_cfg["categorical_features"]:
        if col in df.columns and df[col].isnull().sum() > 0:
            val = df[col].mode()[0]
            df[col] = df[col].fillna(val)
            print(f"[INFO] Imputed '{col}' → mode='{val}'")

    return df

# ── 3. CAP OUTLIERS (IQR) ─────────────────────────────────────────────────────

def cap_outliers(df, feat_cfg):
    for col in feat_cfg["outlier_columns"]:
        if col not in df.columns:
            continue
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        n_clipped = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower, upper)
        print(f"[INFO] Outlier cap '{col}' [{lower:.2f}, {upper:.2f}] → {n_clipped} clipped")
    return df

# ── 4. TRAIN-TEST SPLIT (now happens BEFORE encoding) ─────────────────────────

def split_data(df, cfg, feat_cfg):
    target = feat_cfg["target"]
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["split"]["test_size"],
        stratify=y if cfg["split"]["stratify"] else None,
        random_state=cfg["split"]["random_state"]
    )
    print(f"[INFO] Split → Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test

# ── 5. ENCODE — fit on train ONLY, save encoder.pkl ───────────────────────────

def encode_features(X_train, X_test, feat_cfg):
    """
    CRITICAL: fit both encoders on train only → transform both.
    Saves a single dict of fitted encoder objects so inference.py can
    apply the exact same transformation to a brand-new candidate row,
    including unseen categories (handle_unknown is set to not crash).
    """
    ordinal_maps = feat_cfg.get("ordinal_maps", {})
    ordinal_cols = [c for c in ordinal_maps.keys() if c in X_train.columns]
    onehot_cols  = [c for c in feat_cfg.get("onehot_columns", []) if c in X_train.columns]

    encoders = {}

    # --- Ordinal encoding (fixed category order from features.yaml) ---
    if ordinal_cols:
        categories = [list(ordinal_maps[col].keys()) for col in ordinal_cols]
        ord_enc = OrdinalEncoder(
            categories=categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
        X_train[ordinal_cols] = ord_enc.fit_transform(X_train[ordinal_cols])
        X_test[ordinal_cols]  = ord_enc.transform(X_test[ordinal_cols])
        encoders["ordinal"] = {"encoder": ord_enc, "columns": ordinal_cols}
        print(f"[INFO] Ordinal encoded (fit on train): {ordinal_cols}")

    # --- One-hot encoding ---
    if onehot_cols:
        ohe = OneHotEncoder(
            drop="first",
            handle_unknown="ignore",
            sparse_output=False
        )
        train_ohe = ohe.fit_transform(X_train[onehot_cols])
        test_ohe  = ohe.transform(X_test[onehot_cols])

        ohe_names = ohe.get_feature_names_out(onehot_cols)

        train_ohe_df = pd.DataFrame(train_ohe, columns=ohe_names, index=X_train.index)
        test_ohe_df  = pd.DataFrame(test_ohe,  columns=ohe_names, index=X_test.index)

        X_train = pd.concat([X_train.drop(columns=onehot_cols), train_ohe_df], axis=1)
        X_test  = pd.concat([X_test.drop(columns=onehot_cols),  test_ohe_df],  axis=1)

        encoders["onehot"] = {"encoder": ohe, "columns": onehot_cols, "output_columns": list(ohe_names)}
        print(f"[INFO] One-hot encoded (fit on train): {onehot_cols} → {len(ohe_names)} new columns")

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(encoders, "artifacts/encoder.pkl")
    print("[INFO] Saved fitted encoders → artifacts/encoder.pkl")

    return X_train, X_test, encoders

# ── 6. SCALE — fit on train ONLY ──────────────────────────────────────────────

def scale_features(X_train, X_test, feat_cfg):
    """
    CRITICAL: fit scaler on train only → transform both.
    Fitting on full data before split = data leakage.
    """
    num_cols = [c for c in feat_cfg["numerical_features"] if c in X_train.columns]

    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(scaler, "artifacts/scaler.pkl")
    print(f"[INFO] Scaler fitted on {len(num_cols)} numerical cols → saved artifacts/scaler.pkl")
    return X_train, X_test, scaler

# ── 7. SAVE SPLITS ────────────────────────────────────────────────────────────

def save_splits(X_train, X_test, y_train, y_test, feat_cfg):
    os.makedirs("data/splits", exist_ok=True)
    target = feat_cfg["target"]

    train_df = X_train.copy(); train_df[target] = y_train.values
    test_df  = X_test.copy();  test_df[target]  = y_test.values

    train_df.to_csv("data/splits/train.csv", index=False)
    test_df.to_csv("data/splits/test.csv",   index=False)
    print("[INFO] Saved → data/splits/train.csv & test.csv")

# ── MAIN ──────────────────────────────────────────────────────────────────────

def run_preprocessing():
    cfg      = load_config()
    feat_cfg = load_features()
    mode     = cfg.get("mode", "full").upper()
    print(f"\n{'='*50}")
    print(f"  PREPROCESSING  |  Mode: {mode}")
    print(f"{'='*50}")

    df = load_data()
    df = drop_duplicates(df)
    df = impute_missing(df, feat_cfg)
    df = cap_outliers(df, feat_cfg)

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/cleaned.csv", index=False)
    print("[INFO] Saved → data/processed/cleaned.csv  (pre-encoding, pre-split)")

    X_train, X_test, y_train, y_test = split_data(df, cfg, feat_cfg)
    X_train, X_test, _               = encode_features(X_train, X_test, feat_cfg)
    X_train, X_test, _               = scale_features(X_train, X_test, feat_cfg)
    save_splits(X_train, X_test, y_train, y_test, feat_cfg)

    print(f"\n[DONE] Preprocessing complete | Mode: {mode}\n")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    run_preprocessing()
