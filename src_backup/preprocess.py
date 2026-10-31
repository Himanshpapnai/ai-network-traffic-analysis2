from pathlib import Path

import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

def prepare_features(df):
    """
    Prepare network traffic data for machine learning.

    - Removes the target label to prevent data leakage.
    - Converts usable features to numeric values.
    - Removes columns that contain no usable numeric data.
    - Replaces infinite and missing values.
    - Preserves the original row indices.
    """

    # Work on a copy to avoid modifying the original DataFrame.
    X = df.copy()

    # Remove whitespace from column names.
    X.columns = X.columns.astype(str).str.strip()

    # Remove the target label from the features.
    label_columns = [
        col for col in X.columns
        if col.lower() == "label"
    ]

    X = X.drop(columns=label_columns, errors="ignore")

    # Convert features to numeric values.
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Replace infinite values with missing values.
    X = X.replace([np.inf, -np.inf], np.nan)

    # Remove columns that contain no usable values.
    X = X.dropna(axis=1, how="all")

    # Remove columns with no variation.
    X = X.loc[:, X.nunique(dropna=True) > 1]

    if X.empty or X.shape[1] == 0:
        raise ValueError(
            "No usable numeric features found in the dataset."
        )

    # Fill missing values using each column's median.
    X = X.fillna(X.median())

    # Final validation.
    if X.isna().any().any():
        raise ValueError(
            "Missing values remain after preprocessing."
        )

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(
            "Infinite values remain after preprocessing."
        )

    return X
def inspect_dataset(csv_path):
    """Print basic information about a network traffic CSV."""

    print(f"\nReading dataset: {csv_path.name}")

    # Read only a small sample for initial inspection.
    df = pd.read_csv(
        csv_path,
        nrows=1000,
        low_memory=False
    )

    # Remove accidental spaces in column names.
    df.columns = df.columns.str.strip()

    print("\nDataset shape:", df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst five rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum().sort_values(ascending=False).head(10))

    if "Label" in df.columns:
        print("\nTraffic label distribution:")
        print(df["Label"].value_counts(dropna=False))


def main():
    csv_files = list(RAW_DIR.glob("*.csv"))

    if not csv_files:
        print("No CSV files found in data/raw/.")
        print("Download the dataset and place its CSV files there.")
        return

    inspect_dataset(csv_files[0])


if __name__ == "__main__":
    main()