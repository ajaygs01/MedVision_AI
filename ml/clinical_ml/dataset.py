"""
Dataset loader and leakage-safe preprocessing pipeline for Clinical ML.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ml.clinical_ml.config import (
    DATASET_PATH,
    COLUMN_NAMES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    RANDOM_SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
)


def load_raw_dataset() -> pd.DataFrame:
    """Loads raw UCI Cleveland dataset with missing value parsing."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Clinical dataset not found at: {DATASET_PATH}")
    
    df = pd.read_csv(
        DATASET_PATH,
        names=COLUMN_NAMES,
        na_values="?",
        sep=",",
        skipinitialspace=True,
    )
    
    # Binarize target: 0 = No heart disease, 1-4 = Heart disease present
    df["binary_target"] = (df[TARGET_COLUMN] > 0).astype(int)
    return df


def split_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset into Train (80%), Validation (10%), and Test (10%) splits
    with stratification on the target variable.
    """
    # First split: Train vs Temp (Val + Test)
    temp_ratio = VAL_RATIO + TEST_RATIO
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        random_state=RANDOM_SEED,
        stratify=df["binary_target"],
    )

    # Second split: Val vs Test (50/50 of temp)
    val_ratio_in_temp = VAL_RATIO / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_ratio_in_temp),
        random_state=RANDOM_SEED,
        stratify=temp_df["binary_target"],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs a leakage-free Scikit-Learn ColumnTransformer pipeline:
    - Numerical: Median imputation + Standard Scaling
    - Categorical: Most frequent imputation + OneHotEncoding
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def get_train_val_test_data() -> Dict[str, Any]:
    """Loads, splits, and prepares feature matrices and targets."""
    df = load_raw_dataset()
    train_df, val_df, test_df = split_dataset(df)

    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    
    X_train = train_df[feature_cols]
    y_train = train_df["binary_target"].values

    X_val = val_df[feature_cols]
    y_val = val_df["binary_target"].values

    X_test = test_df[feature_cols]
    y_test = test_df["binary_target"].values

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "feature_cols": feature_cols,
    }


if __name__ == "__main__":
    data = get_train_val_test_data()
    print("Dataset loaded successfully!")
    print(f"Train samples: {len(data['X_train'])} (Positive rate: {data['y_train'].mean():.2%})")
    print(f"Val samples:   {len(data['X_val'])} (Positive rate: {data['y_val'].mean():.2%})")
    print(f"Test samples:  {len(data['X_test'])} (Positive rate: {data['y_test'].mean():.2%})")
