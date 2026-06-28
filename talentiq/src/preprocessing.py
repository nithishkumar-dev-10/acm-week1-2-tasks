import logging
from pathlib import Path
import numpy as np
import pandas as pd
from src.config_loader import load_config, load_features, PROJECT_ROOT


logger = logging.getLogger(__name__)

#loading the raw dataset 
def load_data() -> pd.DataFrame:
    

    cfg = load_config()

    mode = cfg["mode"]
    raw_path = PROJECT_ROOT / cfg["paths"]["raw"][mode]

    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} not found.")

    df = pd.read_csv(raw_path)

    logger.info(f"Loaded dataset ({df.shape[0]} rows, {df.shape[1]} columns)")


    # IBM dataset: map integer columns to string labels for ordinal encoder
    drop_cols = ['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    if 'Attrition' in df.columns:
        df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

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


#Validating the dataset ,checking any missing coloumn and returning the missing coloumn
def validate_data(df: pd.DataFrame) -> None:
    

    features = load_features()

    if df.empty:
        raise ValueError("Dataset is empty.")

    if features["target"] not in df.columns:
        raise ValueError(
            f"Target column '{features['target']}' not found."
        )

    required_columns = (
        features["numerical_features"]
        + features["categorical_features"]
        + [features["target"]]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    logger.info("Dataset validation completed.")

#Handling the missing values , filling the missing values with median for numercial and with mode for categorical coloumns
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    

    features = load_features()

    numerical_columns = features["numerical_features"]
    categorical_columns = features["categorical_features"]

    for column in numerical_columns:
        if df[column].isnull().sum() > 0:
            df[column] = df[column].fillna(df[column].median())

    for column in categorical_columns:
        if df[column].isnull().sum() > 0:
            df[column] = df[column].fillna(df[column].mode()[0])

    logger.info("Missing values handled.")

    return df


#Removing the duplicate rows and store and printing in terminal how many duplicate rows's are dropped 
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    
    before = len(df)

    df = df.drop_duplicates()

    removed = before - len(df)

    logger.info(f"Removed {removed} duplicate rows.")

    return df

#Handling outlier using IQR method
def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    
    features = load_features()

    for column in features["outlier_columns"]:

        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        df[column] = np.clip(df[column], lower, upper)

    logger.info("Outliers handled.")

    return df

#A function to execute all the function in a proper order
def preprocess_data() -> pd.DataFrame:
   
    cfg = load_config()

    mode = cfg["mode"]

    processed_path = PROJECT_ROOT / cfg["paths"]["processed"][mode]

    processed_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_data()

    validate_data(df)

    df = handle_missing_values(df)

    df = remove_duplicates(df)

    df = handle_outliers(df)

    df.to_csv(processed_path, index=False)

    logger.info(f"Saved processed data to {processed_path}")

    return df

#main
if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
    )

    preprocess_data()