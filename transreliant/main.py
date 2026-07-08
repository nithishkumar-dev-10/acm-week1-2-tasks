
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd

from config_loader import load_config
from pipeline import load_cascade_artifacts, predict_cascade


KEY_NUMERIC_FIELDS = [
    "Number of Passengers",
    "Travel Distance",
    "days_before_journey",
    "Seat Availability",
]
KEY_CATEGORICAL_FIELDS = [
    "Class of Travel",
    "Quota",
]


def compute_defaults(cfg: dict) -> dict:

    featured_path = Path(cfg["data"]["featured"])
    if not featured_path.exists():
        raise FileNotFoundError(
            f"{featured_path} not found — run run_pipeline.py at least once first."
        )
    df = pd.read_csv(featured_path)

    numeric_cols = cfg["features"]["numerical"]
    categorical_cols = cfg["features"]["categorical"]
    prompted = set(KEY_NUMERIC_FIELDS + KEY_CATEGORICAL_FIELDS)

    defaults = {}
    for col in numeric_cols:
        if col not in prompted:
            defaults[col] = float(df[col].median())
    for col in categorical_cols:
        if col not in prompted:
            defaults[col] = df[col].mode(dropna=True).iloc[0]

    return defaults


def prompt_numeric(field: str) -> float:
    while True:
        raw = input(f"  {field}: ").strip()
        try:
            return float(raw)
        except ValueError:
            print("    Please enter a number.")


def prompt_categorical(field: str, options: list) -> str:
    options_str = ", ".join(options)
    while True:
        raw = input(f"  {field} [{options_str}]: ").strip()
        if raw in options:
            return raw
        # allow case-insensitive match
        match = next((o for o in options if o.lower() == raw.lower()), None)
        if match:
            return match
        print(f"    Please enter one of: {options_str}")


def collect_input(cfg: dict) -> dict:
    """Prompt for the 6 key fields; category options pulled from featured.csv
    so the CLI only ever offers values that actually exist in the data."""
    featured_path = Path(cfg["data"]["featured"])
    df = pd.read_csv(featured_path)

    print("Enter passenger booking details (press Ctrl+C to quit):\n")

    values = {}
    for field in KEY_NUMERIC_FIELDS:
        values[field] = prompt_numeric(field)
    for field in KEY_CATEGORICAL_FIELDS:
        options = sorted(df[field].dropna().unique().tolist())
        values[field] = prompt_categorical(field, options)

    return values


def build_input_row(cfg: dict, user_values: dict, defaults: dict) -> pd.DataFrame:
    row = {**defaults, **user_values}
    numeric_cols = cfg["features"]["numerical"]
    categorical_cols = cfg["features"]["categorical"]
    ordered = {col: row[col] for col in numeric_cols + categorical_cols}
    return pd.DataFrame([ordered])


def print_result(result_row: pd.Series):
    status = result_row["confirmation_prediction"]
    confidence = result_row["confirmation_confidence"] * 100
    waitlist = result_row["estimated_waitlist_position"]

    print("\n" + "-" * 44)
    print(f"Confirmation Prediction : {status} (confidence {confidence:.0f}%)")
    if pd.notna(waitlist):
        print(f"Estimated Waitlist Pos. : {int(waitlist)} (out of ~200)")
    else:
        print("Estimated Waitlist Pos. : N/A (passenger predicted Confirmed)")
    print("-" * 44 + "\n")


def main():
    cfg = load_config()
    print(f"Loading cascade artifacts (threshold={cfg.get('threshold', 0.5)})...")
    stage1_model, stage1_prep, stage2_model, stage2_prep, threshold = load_cascade_artifacts(cfg)
    defaults = compute_defaults(cfg)
    print("Ready.\n")

    while True:
        try:
            user_values = collect_input(cfg)
        except KeyboardInterrupt:
            print("\nExiting.")
            break

        input_row = build_input_row(cfg, user_values, defaults)
        result = predict_cascade(
            input_row, stage1_model, stage1_prep, stage2_model, stage2_prep, threshold, cfg
        )
        print_result(result.iloc[0])

        again = input("Predict another? [y/N]: ").strip().lower()
        if again != "y":
            break


if __name__ == "__main__":
    main()