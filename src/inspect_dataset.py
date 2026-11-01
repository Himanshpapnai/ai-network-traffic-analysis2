from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
)


def main():
    if not DATASET.exists():
        print(f"Dataset not found: {DATASET}")
        return

    # Read only the header; do not load the 77 MB dataset into memory.
    columns = pd.read_csv(DATASET, nrows=0).columns.tolist()

    print("Dataset:", DATASET.name)
    print("Number of columns:", len(columns))
    print("\nColumn names:")

    for index, column in enumerate(columns, start=1):
        print(f"{index:02d}. {column!r}")


if __name__ == "__main__":
    main()