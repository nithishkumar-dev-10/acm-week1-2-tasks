# src/preprocessing.py
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from config_loader import load_config


def get_feature_lists(cfg: dict) -> tuple[list, list]:
    """
    Pull numeric/categorical feature lists from config.yaml.
    Single source of truth — train_stage1.py and train_stage2.py both
    call this instead of hardcoding column names.
    """
    numeric_cols = cfg["features"]["numerical"]
    categorical_cols = cfg["features"]["categorical"]
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
    """
    numeric   -> StandardScaler
    categorical (incl. booking_urgency_bucket) -> OneHotEncoder(handle_unknown="ignore")
    handle_unknown="ignore" matters here specifically because Stage 2 is fit on a
    smaller subset (Not-Confirmed only) — a category value present at inference
    time but absent from that subset's training fold won't blow up the encoder.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="drop",  # anything not explicitly listed gets dropped, not silently passed through
    )


def build_stage1_preprocessor(cfg: dict) -> ColumnTransformer:
    """Fresh, unfitted preprocessor for Stage 1 (Confirmation Status classifier)."""
    numeric_cols, categorical_cols = get_feature_lists(cfg)
    print(f"Stage 1 preprocessor: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical cols.")
    return build_preprocessor(numeric_cols, categorical_cols)


def build_stage2_preprocessor(cfg: dict) -> ColumnTransformer:
    """
    Fresh, unfitted preprocessor for Stage 2 (Waitlist Position regressor).
    Same feature lists as Stage 1 — the difference is WHERE it gets fit
    (Not-Confirmed-only training rows), not which columns it uses.
    Deliberately a separate instance, not a shared reference.
    """
    numeric_cols, categorical_cols = get_feature_lists(cfg)
    print(f"Stage 2 preprocessor: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical cols.")
    return build_preprocessor(numeric_cols, categorical_cols)


def sanity_check_columns(cfg: dict) -> None:
    """
    Verify every column config.yaml expects actually exists in featured.csv.
    Catches config/data drift (e.g. you rename a feature and forget to
    update config.yaml) before it surfaces as a cryptic sklearn KeyError
    three files downstream in train_stage1.py.
    """
    featured_path = Path(cfg["data"]["featured"])
    if not featured_path.exists():
        raise FileNotFoundError(f"Featured data not found at {featured_path.resolve()} — run feature_engineering.py first.")

    df = pd.read_csv(featured_path)
    numeric_cols, categorical_cols = get_feature_lists(cfg)
    expected = set(numeric_cols + categorical_cols)
    missing = expected - set(df.columns)

    assert not missing, f"config.yaml expects these columns but featured.csv doesn't have them: {missing}"
    print(f"All {len(expected)} configured feature columns present in featured.csv.")


def main():
    cfg = load_config()
    sanity_check_columns(cfg)
    build_stage1_preprocessor(cfg)
    build_stage2_preprocessor(cfg)
    print("Preprocessing module OK — both ColumnTransformers build cleanly.")


if __name__ == "__main__":
    main()