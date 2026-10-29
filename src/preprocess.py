from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


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