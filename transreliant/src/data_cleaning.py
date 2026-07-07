import pandas as pd
import numpy as np
from pathlib import Path
from config_loader import load_config


def load_raw_data(cfg: dict) -> pd.DataFrame:
    """Load the raw ticket confirmation CSV."""
    raw_path = Path(cfg["data"]["raw"])
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found at {raw_path.resolve()}")
    df = pd.read_csv(raw_path)
    df.columns = df.columns.str.strip()
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def drop_leak_and_id_columns(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Drop columns listed in config.drop_columns (IDs + leakage)."""
    cols_to_drop = [c for c in cfg["drop_columns"] if c in df.columns]
    missing = [c for c in cfg["drop_columns"] if c not in df.columns]
    if missing:
        print(f"Warning: these drop_columns were not found in data: {missing}")

    df = df.drop(columns=cols_to_drop)
    print(f"Dropped columns: {cols_to_drop}")
    return df


def derive_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse Date of Journey / Booking Date into datetime, derive:
      - journey_month
      - journey_dayofweek
      - days_before_journey = (journey_date - booking_date).days
    Then drop the two raw date columns (they're not usable as model features directly).
    These three derived columns are required by config.yaml -> features.numerical,
    so this step MUST run before featured.csv / model training.
    """
    if "Date of Journey" not in df.columns or "Booking Date" not in df.columns:
        print("Warning: date columns not found, skipping date feature derivation.")
        return df

    df["Date of Journey"] = pd.to_datetime(df["Date of Journey"], errors="coerce")
    df["Booking Date"] = pd.to_datetime(df["Booking Date"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["Date of Journey", "Booking Date"])
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with unparseable dates.")

    df["journey_month"] = df["Date of Journey"].dt.month
    df["journey_dayofweek"] = df["Date of Journey"].dt.dayofweek
    df["days_before_journey"] = (df["Date of Journey"] - df["Booking Date"]).dt.days

    df = df.drop(columns=["Date of Journey", "Booking Date"])
    print("Derived journey_month, journey_dayofweek, days_before_journey; dropped raw date columns.")
    return df


def encode_targets(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    - Confirmation Status: text -> binary {Confirmed: 1, Not Confirmed: 0}
    - Waitlist Position: strip 'WL' prefix, coerce to numeric.
        * Confirmed rows (status == 1) with missing/0 WL -> 0 (legitimately not waitlisted)
        * Not Confirmed rows with missing WL -> can't train Stage 2 without a label -> drop row
    """
    status_col = cfg["target"]["stage1"]   # "Confirmation Status"
    wl_col = cfg["target"]["stage2"]        # "Waitlist Position"

    df[status_col] = df[status_col].map({"Confirmed": 1, "Not Confirmed": 0})
    before = len(df)
    df = df.dropna(subset=[status_col])
    if before - len(df):
        print(f"Dropped {before - len(df)} rows with unmapped/unknown Confirmation Status.")
    df[status_col] = df[status_col].astype(int)

    df[wl_col] = (
        df[wl_col].astype(str)
        .str.replace("WL", "", regex=False)
        .str.strip()
    )
    df[wl_col] = pd.to_numeric(df[wl_col], errors="coerce")

    # Confirmed passengers legitimately have no waitlist number
    df.loc[df[status_col] == 1, wl_col] = df.loc[df[status_col] == 1, wl_col].fillna(0)

    before = len(df)
    df = df.dropna(subset=[wl_col])  # drops Not-Confirmed rows with missing WL (can't be Stage 2 labels)
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} Not-Confirmed rows with missing Waitlist Position (unusable Stage 2 label).")

    print(f"Encoded targets: {status_col} -> {{0,1}}, {wl_col} -> numeric.")
    return df


def check_and_handle_nulls(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Handle nulls column-by-column instead of blindly dropping rows.
    - 'Special Considerations' nulls mean 'no special consideration' -> fill with 'None', don't drop.
    - Any other unexpected nulls -> drop only those specific rows.
    (Waitlist Position / Confirmation Status nulls are already resolved in encode_targets().)
    """
    null_counts = df.isnull().sum()
    nulls_present = null_counts[null_counts > 0]
    print("Null values found before targeted fill:")
    print(nulls_present if len(nulls_present) > 0 else "None")

    if "Special Considerations" in df.columns:
        # IMPORTANT: don't fill with the literal string "None" — pandas' read_csv
        # treats "None" as a default NA value, so it silently turns back into a
        # null the moment cleaned.csv is saved and re-read (this was the actual
        # bug before, not a broken fillna call). Use a sentinel that isn't in
        # pandas' default na_values list instead.
        df["Special Considerations"] = df["Special Considerations"].astype(object)
        df["Special Considerations"] = df["Special Considerations"].where(
            df["Special Considerations"].notna(), "No_Special_Consideration"
        )

    # Any remaining unexpected nulls in other columns — drop only those rows
    remaining_nulls = df.isnull().sum()
    remaining_nulls = remaining_nulls[remaining_nulls > 0]
    if len(remaining_nulls) > 0:
        print(f"Remaining unexpected nulls, dropping only those rows: {dict(remaining_nulls)}")
        before = len(df)
        df = df.dropna()
        print(f"Dropped {before - len(df)} rows ({before} -> {len(df)})")
    else:
        print("No unexpected nulls remaining after targeted fill.")

    return df


def check_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Report and drop exact duplicate rows."""
    dupe_count = df.duplicated().sum()
    if dupe_count > 0:
        print(f"Found {dupe_count} duplicate rows, dropping them.")
        df = df.drop_duplicates()
    else:
        print("No duplicate rows found.")
    return df


def run_leakage_guardrails(df: pd.DataFrame, cfg: dict) -> None:
    """
    Permanent asserts proving no leakage / no broken targets made it past cleaning.
    Keep these forever — they're what actually prove the claims in your report.
    """
    status_col = cfg["target"]["stage1"]
    wl_col = cfg["target"]["stage2"]

    assert df[status_col].isin([0, 1]).all(), "Confirmation Status has values outside {0,1}"
    assert (df.loc[df[status_col] == 0, wl_col] > 0).all(), \
        "Found a Not-Confirmed row with Waitlist Position <= 0 — label leakage/bug"
    assert df[wl_col].notna().all(), "Waitlist Position still has NaNs after cleaning"
    print("Leakage guardrail asserts passed.")


def save_cleaned_data(df: pd.DataFrame, cfg: dict) -> None:
    """Save cleaned dataframe to the processed data path."""
    out_path = Path(cfg["data"]["cleaned"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved cleaned data to {out_path} — final shape: {df.shape}")


def main():
    cfg = load_config()
    df = load_raw_data(cfg)
    df = drop_leak_and_id_columns(df, cfg)
    df = derive_date_features(df)
    df = encode_targets(df, cfg)
    df = check_and_handle_nulls(df, cfg)
    df = check_duplicates(df)
    run_leakage_guardrails(df, cfg)
    save_cleaned_data(df, cfg)


if __name__ == "__main__":
    main()