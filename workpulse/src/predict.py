import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config_loader import PROJECT_ROOT, load_config, load_features
from src.feature_engineering import engineer_features

logger = logging.getLogger(__name__)


def load_artifacts(model_name: str = "random_forest"):
    """Load the trained model, preprocessor, and feature columns."""
    cfg = load_config()

    preprocessor = joblib.load(PROJECT_ROOT / cfg["paths"]["artifacts"]["preprocessor"])
    feature_columns = joblib.load(PROJECT_ROOT / cfg["paths"]["artifacts"]["feature_columns"])

    model_path = PROJECT_ROOT / cfg["paths"]["models"][model_name]
    model = joblib.load(model_path)

    logger.info(f"Loaded model: {model_name} from {model_path}")

    return model, preprocessor, feature_columns


def clean_raw_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the exact same raw-data cleaning steps used in preprocessing.load_data(),
    so prediction input matches the format the preprocessor was fit on.
    """
    df = df.copy()

    drop_cols = ['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    if 'Attrition' in df.columns:
        df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

    # Cast to string — must match training (OrdinalEncoder was fit on strings)
    for col in ['JobLevel', 'StockOptionLevel']:
        if col in df.columns:
            df[col] = df[col].astype(str)

    col_maps = {
        'Education': {1: 'Below College', 2: 'College', 3: 'Bachelor', 4: 'Master', 5: 'Doctor'},
        'EnvironmentSatisfaction': {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'},
        'JobInvolvement': {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'},
        'JobSatisfaction': {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'},
        'PerformanceRating': {1: 'Low', 2: 'Good', 3: 'Excellent', 4: 'Outstanding'},
        'RelationshipSatisfaction': {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'},
        'WorkLifeBalance': {1: 'Bad', 2: 'Good', 3: 'Better', 4: 'Best'},
    }
    for col, mapping in col_maps.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    return df


def predict_from_dataframe(df: pd.DataFrame, model_name: str = "random_forest") -> pd.DataFrame:
    """
    Run predictions on a raw input dataframe (same raw format as the original CSV).
    Returns the dataframe with two new columns: Prediction and Probability.
    """
    model, preprocessor, feature_columns = load_artifacts(model_name)
    feat = load_features()
    target = feat["target"]

    # Step 1 — clean raw input exactly like preprocessing.load_data()
    df_clean = clean_raw_input(df)

    if target in df_clean.columns:
        df_clean = df_clean.drop(columns=[target])

    # Step 2 — feature engineering, same as training
    df_input = engineer_features(df_clean)

    # Step 3 — align columns to match training feature order
    missing_cols = set(feature_columns) - set(df_input.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in input: {missing_cols}")
    df_input = df_input[feature_columns]

    # Step 4 — transform using the fitted preprocessor
    X = preprocessor.transform(df_input)

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    result = df.copy()
    result["Prediction"] = np.where(predictions == 1, "Yes", "No")
    result["Probability"] = np.round(probabilities, 4)

    return result


# ---------------------------------------------------------------------------
# Interactive single-employee input (CLI)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Interactive single-employee input (CLI) — minimal-input mode
# ---------------------------------------------------------------------------
# Only the fields below are asked of the user. Everything else the model
# needs is auto-filled using the median (numeric) / mode (categorical or
# ordinal) computed from the raw training CSV, so a non-HR user isn't
# expected to know internal fields like DailyRate or EnvironmentSatisfaction.

ASK_NUMERICAL_FIELDS = ["Age", "MonthlyIncome", "YearsAtCompany"]

ASK_CATEGORICAL_OPTIONS = {
    "Department": ["Sales", "Research & Development", "Human Resources"],
    "JobRole": ["Sales Executive", "Research Scientist", "Laboratory Technician",
                "Manufacturing Director", "Healthcare Representative", "Manager",
                "Sales Representative", "Research Director", "Human Resources"],
    "MaritalStatus": ["Single", "Married", "Divorced"],
    "OverTime": ["Yes", "No"],
}

# All raw columns the pipeline needs end-to-end (numerical_features +
# categorical_features + ordinal_maps keys from features.yaml).
ALL_NUMERICAL_FIELDS = [
    "Age", "DailyRate", "DistanceFromHome", "HourlyRate", "MonthlyIncome",
    "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike", "TotalWorkingYears",
    "TrainingTimesLastYear", "YearsAtCompany", "YearsInCurrentRole",
    "YearsSinceLastPromotion", "YearsWithCurrManager",
]
ALL_CATEGORICAL_FIELDS = [
    "BusinessTravel", "Department", "EducationField", "Gender", "JobRole",
    "MaritalStatus", "OverTime",
]
# Ordinal fields stored as raw ints in the CSV (clean_raw_input() maps
# Education/EnvironmentSatisfaction/etc. ints -> labels; JobLevel/
# StockOptionLevel stay as the int, cast to str later).
ALL_ORDINAL_FIELDS = [
    "Education", "EnvironmentSatisfaction", "JobInvolvement", "JobLevel",
    "JobSatisfaction", "PerformanceRating", "RelationshipSatisfaction",
    "StockOptionLevel", "WorkLifeBalance",
]


def _load_raw_training_df() -> pd.DataFrame:
    cfg = load_config()
    mode = cfg["mode"]
    raw_path = PROJECT_ROOT / cfg["paths"]["raw"][mode]
    return pd.read_csv(raw_path)


def compute_defaults() -> dict:
    """Compute median (numeric/ordinal) and mode (categorical) defaults
    from the raw training CSV for every field not asked of the user."""
    raw = _load_raw_training_df()
    defaults = {}

    for field in ALL_NUMERICAL_FIELDS:
        if field not in ASK_NUMERICAL_FIELDS:
            defaults[field] = float(raw[field].median())

    for field in ALL_CATEGORICAL_FIELDS:
        if field not in ASK_CATEGORICAL_OPTIONS:
            defaults[field] = raw[field].mode()[0]

    for field in ALL_ORDINAL_FIELDS:
        defaults[field] = int(raw[field].median())

    return defaults


def _prompt_float(field: str) -> float:
    while True:
        raw = input(f"{field}: ").strip()
        try:
            return float(raw)
        except ValueError:
            print("  Please enter a numeric value.")


def _prompt_choice(field: str, options: list) -> str:
    print(f"{field}:")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        raw = input(f"Select {field} [1-{len(options)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print("  Invalid choice, try again.")


def collect_user_input() -> pd.DataFrame:
    """Prompt the user for only the easy-to-know fields; auto-fill the rest
    using training-data medians/modes."""
    print("\n--- Enter employee details for attrition prediction ---")
    print(f"(Just {len(ASK_NUMERICAL_FIELDS) + len(ASK_CATEGORICAL_OPTIONS)} quick questions — "
          "everything else is auto-filled from typical values)\n")

    record = {}

    for field in ASK_NUMERICAL_FIELDS:
        record[field] = _prompt_float(field)

    for field, options in ASK_CATEGORICAL_OPTIONS.items():
        record[field] = _prompt_choice(field, options)

    defaults = compute_defaults()
    record.update(defaults)

    print()
    return pd.DataFrame([record])


def predict_interactive(model_name: str = "random_forest") -> pd.DataFrame:
    """Collect a single employee's details (minimal input) via CLI prompts and predict."""
    df = collect_user_input()
    result = predict_from_dataframe(df, model_name)
    return result


def predict_from_csv(csv_path: str, model_name: str = "random_forest", output_path: str = None) -> pd.DataFrame:
    """Run predictions on a CSV file and optionally save the results."""
    df = pd.read_csv(csv_path)
    result = predict_from_dataframe(df, model_name)

    if output_path:
        result.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")

    return result


def run_prediction(input_path: str = None, model_name: str = "random_forest", output_path: str = None) -> None:
    """Entry point used by main.py --stage predict.

    If input_path is None or "interactive", prompts the user on the CLI
    for a single employee's details. Otherwise treats input_path as a CSV
    file and runs batch prediction.
    """
    if input_path is None or input_path.lower() == "interactive":
        result = predict_interactive(model_name)
        if output_path:
            result.to_csv(output_path, index=False)
            logger.info(f"Predictions saved to {output_path}")
        logger.info(f"Predicted {len(result)} row using {model_name}.")
        print(result[["Prediction", "Probability"]].to_string(index=False))
        return

    if output_path is None:
        output_path = str(PROJECT_ROOT / "reports" / "predictions.csv")

    result = predict_from_csv(input_path, model_name, output_path)
    logger.info(f"Predicted {len(result)} rows using {model_name}.")
    print(result[["Prediction", "Probability"]].head(10).to_string(index=False))