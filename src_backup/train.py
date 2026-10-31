import json
from pathlib import Path
from xml.parsers.expat import model

import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from preprocess import prepare_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "isolation_forest.joblib"


def find_label_column(df):
    """Find the label column without relying on capitalization."""

    for column in df.columns:
        if column.strip().lower() == "label":
            return column

    return None


def is_benign(labels):
    """Identify benign labels in CICIDS-style datasets."""

    normalized = (
        labels.astype(str)
        .str.strip()
        .str.lower()
    )

    return normalized.isin({
        "benign",
        "normal",
    })


def load_dataset(csv_path):
    """Load data, retaining original labels separately."""

    df = pd.read_csv(
        csv_path,
        low_memory=False
    )

    df.columns = df.columns.str.strip()

    label_column = find_label_column(df)

    if label_column is None:
        raise ValueError(
            "The CSV must contain a Label column "
            "for benign-only training and evaluation."
        )

    # Create model features without using the labels.
    X = prepare_features(df)

    # Keep labels separately for sampling and evaluation.
    labels = df.loc[X.index, label_column]

    return X, labels


def main():
    csv_files = sorted(RAW_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            "Place a labelled network traffic CSV in data/raw/."
        )

    csv_path = csv_files[0]

    print(f"Loading dataset: {csv_path.name}")

    X, labels = load_dataset(csv_path)

    benign_mask = is_benign(labels)

    X_benign = X.loc[benign_mask]

    if len(X_benign) < 100:
        raise ValueError(
            "Too few benign rows for a meaningful initial model."
        )

    # Keep training manageable on a typical laptop.
    max_training_rows = 100_000

    if len(X_benign) > max_training_rows:
        X_benign = X_benign.sample(
            n=max_training_rows,
            random_state=42
        )

    print("\nTraining details")
    print("----------------")
    print("Total usable rows:", len(X))
    print("Benign training rows:", len(X_benign))
    print("Feature count:", X.shape[1])

    # Train only on benign traffic.
    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_benign)

    # Save the exact feature order with the model.
    artifact = {
        "model": model,
        "feature_columns": X.columns.tolist(),
        "feature_medians": X_benign.median().to_dict(),
        "training_rows": len(X_benign),
        "training_source": csv_path.name,
    }

    joblib.dump(artifact, MODEL_PATH)

    print("\nModel saved to:", MODEL_PATH)

    # Evaluate against the original labels.
    # Here, 1 = benign and 0 = attack.
    y_true = benign_mask.astype(int)

    # Model prediction: 1 = inlier, -1 = outlier.
    predictions = model.predict(X)

    # Convert model outputs to the evaluation convention.
    y_pred = (predictions == 1).astype(int)

    metrics = {
        "rows_evaluated": int(len(X)),
        "features": int(X.shape[1]),
        "benign_training_rows": int(len(X_benign)),
        "accuracy": float((y_true == y_pred).mean()),
        "benign_precision": float(
            precision_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0
            )
        ),
        "benign_recall": float(
            recall_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0
            )
        ),
        "benign_f1": float(
            f1_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0
            )
        ),
        "attack_precision": float(
            precision_score(
                y_true,
                y_pred,
                pos_label=0,
                zero_division=0
            )
        ),
        "attack_recall": float(
            recall_score(
                y_true,
                y_pred,
                pos_label=0,
                zero_division=0
            )
        ),
        "attack_f1": float(
            f1_score(
                y_true,
                y_pred,
                pos_label=0,
                zero_division=0
            )
        ),
    }

    print("\nConfusion matrix")
    print(
        confusion_matrix(
            y_true,
            y_pred,
            labels=[1, 0]
        )
    )

    print("\nClassification report")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[1, 0],
            target_names=["Benign", "Attack"],
            zero_division=0
        )
    )

    report_path = REPORT_DIR / "model_metrics.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\nMetrics saved to:", report_path)


if __name__ == "__main__":
    main()