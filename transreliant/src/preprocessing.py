# src/preprocessing.py
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from config_loader import load_config, get_path


def get_feature_lists(cfg: dict) -> tuple[list, list]:
    
    numeric_cols = cfg["features"]["numerical"]
    categorical_cols = cfg["features"]["categorical"]
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
    
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="drop",  # anything not explicitly listed gets dropped, not silently passed through
    )


def build_stage1_preprocessor(cfg: dict) -> ColumnTransformer:
    
    numeric_cols, categorical_cols = get_feature_lists(cfg)
    print(f"Stage 1 preprocessor: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical cols.")
    return build_preprocessor(numeric_cols, categorical_cols)


def build_stage2_preprocessor(cfg: dict) -> ColumnTransformer:
   
    numeric_cols, categorical_cols = get_feature_lists(cfg)
    print(f"Stage 2 preprocessor: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical cols.")
    return build_preprocessor(numeric_cols, categorical_cols)


def sanity_check_columns(cfg: dict) -> None:
 
    featured_path = Path(get_path(cfg, "data", "featured"))
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