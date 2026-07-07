# src/feature_engineering.py
import pandas as pd
import numpy as np
from pathlib import Path
from config_loader import load_config


def load_cleaned_data(cfg: dict) -> pd.DataFrame:
    """Load the cleaned CSV produced by data_cleaning.py."""
    cleaned_path = Path(cfg["data"]["cleaned"])
    if not cleaned_path.exists():
        raise FileNotFoundError(f"Cleaned data not found at {cleaned_path.resolve()} — run data_cleaning.py first.")
    df = pd.read_csv(cleaned_path)
    print(f"Loaded cleaned data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def add_peak_holiday_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Binarize Holiday or Peak Season: Yes/No -> 1/0."""
    df["is_peak_or_holiday"] = (df["Holiday or Peak Season"] == "Yes").astype(int)
    print("Added is_peak_or_holiday.")
    return df


def add_seat_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """
    seat_pressure = Number of Passengers / (Seat Availability + 1)
    Strongest engineered signal — approximates booking pressure.
    +1 in denominator avoids div-by-zero when Seat Availability == 0.
    """
    df["seat_pressure"] = df["Number of Passengers"] / (df["Seat Availability"] + 1)
    print("Added seat_pressure.")
    return df


def add_route_length_per_stop(df: pd.DataFrame) -> pd.DataFrame:
    """route_length_per_stop = Travel Distance / (Number of Stations + 1)"""
    df["route_length_per_stop"] = df["Travel Distance"] / (df["Number of Stations"] + 1)
    print("Added route_length_per_stop.")
    return df


def add_booking_urgency_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin days_before_journey into last_minute / short / planned / early.
    Captures the nonlinear 'last-minute booking = high waitlist risk' pattern.
    """
    df["booking_urgency_bucket"] = pd.cut(
        df["days_before_journey"],
        bins=[-1, 2, 14, 59, np.inf],
        labels=["last_minute", "short", "planned", "early"],
    )
    print("Added booking_urgency_bucket.")
    return df


def sanity_check(df: pd.DataFrame) -> None:
    """Quick guardrails — new columns shouldn't introduce NaNs or infs."""
    new_cols = ["is_peak_or_holiday", "seat_pressure", "route_length_per_stop", "booking_urgency_bucket"]
    nulls = df[new_cols].isnull().sum()
    assert nulls.sum() == 0, f"NaNs introduced in engineered features:\n{nulls[nulls > 0]}"

    numeric_new = ["seat_pressure", "route_length_per_stop"]
    inf_check = np.isinf(df[numeric_new]).sum()
    assert inf_check.sum() == 0, f"Infs introduced in engineered features:\n{inf_check[inf_check > 0]}"
    print("Feature engineering sanity checks passed.")


def save_featured_data(df: pd.DataFrame, cfg: dict) -> None:
    out_path = Path(cfg["data"]["featured"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved featured data to {out_path} — final shape: {df.shape}")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Single entry point — used by run_pipeline.py too."""
    df = add_peak_holiday_flag(df)
    df = add_seat_pressure(df)
    df = add_route_length_per_stop(df)
    df = add_booking_urgency_bucket(df)
    return df


def main():
    cfg = load_config()
    df = load_cleaned_data(cfg)
    df = add_features(df)
    sanity_check(df)
    save_featured_data(df, cfg)


if __name__ == "__main__":
    main()