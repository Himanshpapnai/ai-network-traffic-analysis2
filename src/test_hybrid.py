from pathlib import Path

import pandas as pd

from src.predict import predict_traffic


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def main():

    csv_files = sorted(
        RAW_DIR.glob("*.csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            "No CSV files found in data/raw/."
        )

    csv_path = csv_files[0]

    print("Reading:", csv_path.name)

    # Load a small sample to avoid processing the whole file.
    df = pd.read_csv(
        csv_path,
        nrows=1000,
        low_memory=False
    )

    results = predict_traffic(df)

    print("\nHybrid Model Results")
    print("--------------------")

    print(
        "Selected strategy:",
        results["selected_strategy"].iloc[0]
    )

    print(
        "Total flows:",
        len(results)
    )

    print(
        "Normal flows:",
        int(
            (results["status"] == "Normal").sum()
        )
    )

    print(
        "Potential anomalies:",
        int(
            (
                results["status"]
                == "Potential Anomaly"
            ).sum()
        )
    )

    print("\nFirst 10 predictions:")

    print(
        results[
            [
                "rf_attack_probability",
                "if_anomaly_score",
                "rf_prediction",
                "if_prediction",
                "prediction",
                "status",
            ]
        ].head(10).to_string(index=False)
    )

    output_path = (
        REPORT_DIR / "terminal_hybrid_results.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    print(
        "\nSaved results to:",
        output_path
    )


if __name__ == "__main__":
    main()