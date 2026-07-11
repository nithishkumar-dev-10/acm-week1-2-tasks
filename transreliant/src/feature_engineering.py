# src/feature_engineering.py
import pandas as pd
import numpy as np
from pathlib import Path
from config_loader import load_config, get_path


def load_cleaned_data(cfg: dict) -> pd.DataFrame:
 
    cleaned_path = Path(get_path(cfg, "data", "cleaned"))
    if not cleaned_path.exists():
        raise FileNotFoundError(f"Cleaned data not found at {cleaned_path.resolve()} — run data_cleaning.py first.")
    df = pd.read_csv(cleaned_path)
    print(f"Loaded cleaned data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def add_peak_holiday_flag(df: pd.DataFrame) -> pd.DataFrame:

    df["is_peak_or_holiday"] = (df["Holiday or Peak Season"] == "Yes").astype(int)
    print("Added is_peak_or_holiday.")
    return df


def add_seat_pressure(df: pd.DataFrame) -> pd.DataFrame:
   
    df["seat_pressure"] = df["Number of Passengers"] / (df["Seat Availability"] + 1)
    print("Added seat_pressure.")
    return df


def add_route_length_per_stop(df: pd.DataFrame) -> pd.DataFrame:
    
    df["route_length_per_stop"] = df["Travel Distance"] / (df["Number of Stations"] + 1)
    print("Added route_length_per_stop.")
    return df


def add_booking_urgency_bucket(df: pd.DataFrame) -> pd.DataFrame:
    
    df["booking_urgency_bucket"] = pd.cut(
        df["days_before_journey"],
        bins=[-1, 2, 14, 59, np.inf],
        labels=["last_minute", "short", "planned", "early"],
    )
    print("Added booking_urgency_bucket.")
    return df


def add_route_pair(df: pd.DataFrame) -> pd.DataFrame:
    df["route_pair"] = df["Source Station"] + "_" + df["Destination Station"]
    print("Added route_pair.")
    return df


def sanity_check(df: pd.DataFrame) -> None:
    
    new_cols = ["is_peak_or_holiday", "seat_pressure", "route_length_per_stop", "booking_urgency_bucket"]
    nulls = df[new_cols].isnull().sum()
    assert nulls.sum() == 0, f"NaNs introduced in engineered features:\n{nulls[nulls > 0]}"

    numeric_new = ["seat_pressure", "route_length_per_stop"]
    inf_check = np.isinf(df[numeric_new]).sum()
    assert inf_check.sum() == 0, f"Infs introduced in engineered features:\n{inf_check[inf_check > 0]}"
    print("Feature engineering sanity checks passed.")


def save_featured_data(df: pd.DataFrame, cfg: dict) -> None:
    out_path = Path(get_path(cfg, "data", "featured"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved featured data to {out_path} — final shape: {df.shape}")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
   
    df = add_peak_holiday_flag(df)
    df = add_seat_pressure(df)
    df = add_route_length_per_stop(df)
    df = add_booking_urgency_bucket(df)
    df = add_route_pair(df)
    return df


def main():
    cfg = load_config()
    df = load_cleaned_data(cfg)
    df = add_features(df)
    sanity_check(df)
    save_featured_data(df, cfg)


if __name__ == "__main__":
    main()