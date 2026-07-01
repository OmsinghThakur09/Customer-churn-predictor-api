# src/preprocess.py

"""
Reusable preprocessing pipeline for the Churn Predictor.
Two ways this file gets used:
1. IMPORTED   -> train.py and predict.py import prepare_dataframe(), load_preprocessor(),
                 get_feature_names()
2. RUN DIRECTLY -> `python src/preprocess.py` executes the FULL pipeline:
                    raw CSV -> clean -> split -> fit -> SMOTE -> save artifacts.
                    This is what you run first, on any new dataset, before train.py.

Usage:
    python src/preprocess.py
    python src/preprocess.py --input data/raw/some_new_export.csv
"""

import argparse
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "telco-churn.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

# Column definitions
NUMERIC_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']

CATEGORICAL_FEATURES = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

COLUMNS_TO_DROP = ['customerID']
TARGET_COLUMN = 'Churn'
TARGET_MAP = {'Yes': 1, 'No': 0}


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    TotalCharges arrives as a string with blank spaces for brand-new customers
    Churn target arrives as 'Yes'/'No' text
    """
    df = df.copy()

    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(0)

    if TARGET_COLUMN in df.columns and df[TARGET_COLUMN].dtype == object:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map(TARGET_MAP)

    return df


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a raw (or single-customer) DataFrame and return X, ready for the
    preprocessor. Steps: drop ID -> collapse "No x service" categories -> drop target.
    Used at BOTH training time (on the full dataset) and prediction time
    (on a single-row DataFrame built from a customer dict).
    """
    df = df.copy()
    df = df.drop(columns=COLUMNS_TO_DROP, errors='ignore')

    internet_cols = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]

    for col in internet_cols:
        if col in df.columns:
            df[col] = df[col].replace('No internet service', 'No')
    if 'MultipleLines' in df.columns:
        df['MultipleLines'] = df['MultipleLines'].replace('No phone service', 'No')

    X = df.drop(columns=[TARGET_COLUMN], errors='ignore')
    return X


def build_preprocessor() -> ColumnTransformer:
    """
    Build and return an unfitted ColumnTransformer.
    Call .fit_transform(X_train) to fit it.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(
            drop='if_binary',
            handle_unknown='ignore',
            sparse_output=False
        ))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, NUMERIC_FEATURES),
        ('cat', categorical_transformer, CATEGORICAL_FEATURES)
    ])

    return preprocessor


def get_feature_names(fitted_preprocessor: ColumnTransformer) -> list:
    """
    Return human-readable feature names after OHE expansion.
    Required for SHAP .
    :param fitted_preprocessor:
    :return: list of feature names
    """
    ohe_names = fitted_preprocessor \
        .named_transformers_['cat']['onehot'] \
        .get_feature_names_out(CATEGORICAL_FEATURES).tolist()

    return NUMERIC_FEATURES + ohe_names


def save_preprocessor(preprocessor: ColumnTransformer, path: str) -> None:
    """Save a fitted preprocessor to disk."""
    joblib.dump(preprocessor, path)
    print(f"Preprocessor saved → {path}")


def load_preprocessor(path: str) -> ColumnTransformer:
    """Load a fitted preprocessor from disk."""
    return joblib.load(path)


def run_preprocessing(input_path: Path = RAW_DATA_PATH,
                      output_dir: Path = PROCESSED_DIR,
                      model_dir: Path = MODELS_DIR,
                      test_size: float = 0.2,
                      random_state: int = 42) -> None:
    """
    The full, run-once-per-dataset pipeline:
    raw CSV -> clean -> split -> fit preprocessor -> SMOTE (train only) -> save.

    After this runs, train.py can just load the .npy files and train.
    Re-run this any time you have a NEW raw dataset you want to train on.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"Loading raw data from {input_path}...")
    df = pd.read_csv(input_path)
    df = clean_raw_data(df)

    if df[TARGET_COLUMN].isnull().any():
        raise ValueError(
            "Found rows with missing/unmapped Churn values after clean_raw_data(). "
            "Check that the target column only contains 'Yes'/'No' or 0/1."
        )

    y = df[TARGET_COLUMN]
    X = prepare_dataframe(df)

    print(f"Dataset shape: {X.shape} | Churn rate: {y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

    print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = get_feature_names(preprocessor)
    print(f"Encoded feature count: {len(feature_names)}")

    print("Applying SMOTE to training data only (test set stays real-world)...")
    smote = SMOTE(random_state=random_state)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train)
    print(f"Train churn rate before SMOTE: {y_train.mean():.1%} -> after: {y_train_balanced.mean():.1%}")

    # Save everything train.py will need
    np.save(output_dir / "X_train.npy", X_train_balanced)
    np.save(output_dir / "X_test.npy", X_test_processed)
    np.save(output_dir / "y_train.npy", y_train_balanced)
    np.save(output_dir / "y_test.npy", y_test.values)
    pd.Series(feature_names).to_csv(output_dir / "feature_names.csv", header=["features"], index=False)
    save_preprocessor(preprocessor, str(MODELS_DIR / "preprocessor.pkl"))

    print("\n✅ Preprocessing complete. Saved:")
    for f in ["X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy", "feature_names.csv"]:
        print(f"  data/processed/{f}")
    print("  models/preprocessor.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full churn-data preprocessing pipeline.")
    parser.add_argument("--input", type=str, default=str(RAW_DATA_PATH),
                        help="Path to raw CSV (default: data/raw/telco-churn.csv)")
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    run_preprocessing(input_path=Path(args.input), test_size=args.test_size)
