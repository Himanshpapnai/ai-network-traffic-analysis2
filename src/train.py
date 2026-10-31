import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "hybrid_model.joblib"


# --------------------------------------------------
# 2. CONFIGURATION
# --------------------------------------------------

RANDOM_STATE = 42

MAX_TRAINING_ROWS = 100_000

# Calibrate the anomaly threshold so that approximately
# 5% of benign validation flows score above it.
IF_BENIGN_QUANTILE = 95

# Candidate thresholds for Random Forest attack probability.
RF_THRESHOLDS = np.arange(0.10, 0.91, 0.05)


# --------------------------------------------------
# 3. LOAD AND PREPARE THE RAW DATA
# --------------------------------------------------

def find_label_column(df):
    """Find the Label column regardless of capitalization."""

    for column in df.columns:
        if column.strip().lower() == "label":
            return column

    raise ValueError(
        "The dataset must contain a Label column."
    )


def load_dataset():
    """Load the first CSV in data/raw/."""

    csv_files = sorted(RAW_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            "No CSV files found in data/raw/. "
            "Place a labelled network traffic CSV there."
        )

    csv_path = csv_files[0]

    print(f"Loading dataset: {csv_path.name}")

    df = pd.read_csv(
        csv_path,
        low_memory=False,
    )

    df.columns = df.columns.str.strip()

    df = df.loc[:, ~df.columns.duplicated()]

    label_column = find_label_column(df)

    # Drop rows with missing labels.
    df = df.dropna(subset=[label_column]).copy()

    # Normalize the label values.
    normalized_labels = (
        df[label_column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Binary classification:
    # 0 = benign
    # 1 = attack
    benign_labels = {
        "benign",
        "normal",
    }

    y = (~normalized_labels.isin(benign_labels)).astype(int)

    # Exclude labels and common identifiers from model features.
    excluded_columns = {
        "label",
        "attack",
        "class",
        "target",
        "flow id",
        "flowid",
        "source ip",
        "src ip",
        "destination ip",
        "dst ip",
        "timestamp",
        "simillarhttp",
    }

    feature_columns = [
        column
        for column in df.columns
        if column.strip().lower() not in excluded_columns
    ]

    # CICIDS-style flow datasets generally contain numeric features.
    # Non-numeric feature columns are converted to missing values.
    X = df[feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Remove rows where every candidate feature is missing.
    usable_rows = X.notna().any(axis=1)

    X = X.loc[usable_rows].copy()
    y = y.loc[usable_rows].copy()

    if y.nunique() < 2:
        raise ValueError(
            "The dataset must contain both benign and attack examples."
        )

    if not (y == 0).any():
        raise ValueError(
            "No benign examples found. Check the dataset labels."
        )

    if not (y == 1).any():
        raise ValueError(
            "No attack examples found. Check the dataset labels."
        )

    print("\nDataset information")
    print("-------------------")
    print("Usable rows:", len(X))
    print("Candidate features:", len(X.columns))
    print("Benign rows:", int((y == 0).sum()))
    print("Attack rows:", int((y == 1).sum()))

    return X, y, csv_path.name


# --------------------------------------------------
# 4. TRAINING-ONLY FEATURE PREPROCESSING
# --------------------------------------------------

def prepare_splits(X_train, X_val, X_test):
    """
    Select usable features and calculate medians using
    training data only.
    """

    # Retain features with at least one valid training value
    # and more than one distinct training value.
    feature_columns = [
        column
        for column in X_train.columns
        if X_train[column].notna().any()
        and X_train[column].nunique(dropna=True) > 1
    ]

    if not feature_columns:
        raise ValueError(
            "No usable numeric features remain in the training set."
        )

    X_train = X_train[feature_columns].copy()
    X_val = X_val.reindex(columns=feature_columns).copy()
    X_test = X_test.reindex(columns=feature_columns).copy()

    # Calculate medians using the training data only.
    medians = X_train.median()

    X_train = X_train.fillna(medians)
    X_val = X_val.fillna(medians)
    X_test = X_test.fillna(medians)

    # Guard against extreme values that cannot be represented
    # safely as float32.
    max_float32 = np.finfo(np.float32).max

    valid_columns = [
        column
        for column in feature_columns
        if np.isfinite(
            X_train[column].to_numpy(dtype=np.float64)
        ).all()
        and (
            X_train[column].abs() <= max_float32
        ).all()
    ]

    X_train = X_train[valid_columns]
    X_val = X_val[valid_columns]
    X_test = X_test[valid_columns]

    medians = medians[valid_columns]

    if not valid_columns:
        raise ValueError(
            "No features remain after numerical validation."
        )

    if not np.isfinite(
        X_train.to_numpy(dtype=np.float64)
    ).all():
        raise ValueError(
            "Training features contain invalid values."
        )

    return (
        X_train,
        X_val,
        X_test,
        valid_columns,
        medians.to_dict(),
    )


# --------------------------------------------------
# 5. METRIC CALCULATION
# --------------------------------------------------

def calculate_metrics(y_true, y_pred):
    """Calculate binary attack-detection metrics."""

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "false_positive_rate": float(
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0.0
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "confusion_matrix": matrix.tolist(),
    }


# --------------------------------------------------
# 6. SELECT A RANDOM FOREST THRESHOLD
# --------------------------------------------------

def select_rf_threshold(y_val, probabilities):
    """
    Choose the probability threshold with the best validation F1.

    This uses validation labels, never test labels.
    """

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in RF_THRESHOLDS:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_val,
            predictions,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)

    return best_threshold


# --------------------------------------------------
# 7. TRAIN BOTH ALGORITHMS
# --------------------------------------------------

def main():

    X, y, source_name = load_dataset()

    # Split 1:
    # 80% development data, 20% final test data.
    X_development, X_test, y_development, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    # Split 2:
    # From the 80% development data, allocate:
    # 60% of the full dataset to training,
    # 20% to validation,
    # 20% to final testing.
    X_train, X_val, y_train, y_val = train_test_split(
        X_development,
        y_development,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_development,
    )

    # Optionally limit training rows for a first experiment.
    # Sampling happens only after the train/validation/test split.
    if len(X_train) > MAX_TRAINING_ROWS:

        rng = np.random.RandomState(RANDOM_STATE)

        selected_indices = []

        for class_label in [0, 1]:

            class_indices = y_train[
                y_train == class_label
            ].index.to_numpy()

            class_count = int(
                MAX_TRAINING_ROWS
                * len(class_indices)
                / len(y_train)
            )

            class_count = max(
                1,
                min(class_count, len(class_indices)),
            )

            selected_indices.extend(
                rng.choice(
                    class_indices,
                    size=class_count,
                    replace=False,
                ).tolist()
            )

        selected_indices = sorted(set(selected_indices))

        X_train = X_train.loc[selected_indices]
        y_train = y_train.loc[selected_indices]

    (
        X_train,
        X_val,
        X_test,
        feature_columns,
        feature_medians,
    ) = prepare_splits(
        X_train,
        X_val,
        X_test,
    )

    print("\nSplit information")
    print("-----------------")
    print("Training rows:", len(X_train))
    print("Validation rows:", len(X_val))
    print("Test rows:", len(X_test))
    print("Selected features:", len(feature_columns))

    # ----------------------------------------------
    # 7A. RANDOM FOREST: SUPERVISED CLASSIFICATION
    # ----------------------------------------------

    print("\nTraining Random Forest...")

    rf_model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=2,
    )

    rf_model.fit(
        X_train,
        y_train,
    )

    # Obtain the probability of the attack class.
    attack_class_index = list(
        rf_model.classes_
    ).index(1)

    rf_val_probabilities = rf_model.predict_proba(
        X_val
    )[:, attack_class_index]

    rf_threshold = select_rf_threshold(
        y_val,
        rf_val_probabilities,
    )

    print(
        "Selected RF probability threshold:",
        round(rf_threshold, 3),
    )

    # ----------------------------------------------
    # 7B. ISOLATION FOREST: UNSUPERVISED DETECTION
    # ----------------------------------------------

    print("\nTraining Isolation Forest...")

    # Only benign training observations are used.
    benign_training_mask = (
        y_train.to_numpy() == 0
    )

    X_train_benign = X_train.loc[
        benign_training_mask
    ]

    if len(X_train_benign) < 100:
        raise ValueError(
            "Too few benign training examples for this experiment."
        )

    if_model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    if_model.fit(
        X_train_benign
    )

    # A larger score means more anomalous.
    if_val_scores = -if_model.score_samples(
        X_val
    )

    # Calibrate the threshold using BENIGN validation traffic only.
    benign_validation_scores = if_val_scores[
        y_val.to_numpy() == 0
    ]

    if len(benign_validation_scores) == 0:
        raise ValueError(
            "Validation set contains no benign observations."
        )

    if_threshold = float(
        np.percentile(
            benign_validation_scores,
            IF_BENIGN_QUANTILE,
        )
    )

    print(
        "Selected IF anomaly threshold:",
        round(if_threshold, 6),
    )

    # ----------------------------------------------
    # 7C. COMPARE DECISION RULES ON VALIDATION DATA
    # ----------------------------------------------

    rf_val_pred = (
        rf_val_probabilities >= rf_threshold
    ).astype(int)

    if_val_pred = (
        if_val_scores >= if_threshold
    ).astype(int)

    # Hybrid OR:
    # Flag if either model identifies an anomaly.
    hybrid_or_val = (
        (rf_val_pred == 1)
        | (if_val_pred == 1)
    ).astype(int)

    # Hybrid AND:
    # Flag only if both models identify an anomaly.
    hybrid_and_val = (
        (rf_val_pred == 1)
        & (if_val_pred == 1)
    ).astype(int)

    validation_candidates = {
        "random_forest": rf_val_pred,
        "hybrid_or": hybrid_or_val,
        "hybrid_and": hybrid_and_val,
    }

    validation_metrics = {}

    for name, predictions in validation_candidates.items():

        metrics = calculate_metrics(
            y_val,
            predictions,
        )

        validation_metrics[name] = metrics

        print(
            f"\nValidation {name}: "
            f"F1={metrics['f1_score']:.4f}, "
            f"Precision={metrics['precision']:.4f}, "
            f"Recall={metrics['recall']:.4f}, "
            f"FPR={metrics['false_positive_rate']:.4f}"
        )

    # Select the strategy with the highest validation F1.
    # Python preserves insertion order, so ties favor RF,
    # then OR, then AND.
    selected_strategy = max(
        validation_metrics,
        key=lambda name: validation_metrics[name]["f1_score"],
    )

    print(
        "\nSelected decision strategy:",
        selected_strategy,
    )

    # ----------------------------------------------
    # 7D. FINAL EVALUATION ON HELD-OUT TEST DATA
    # ----------------------------------------------

    rf_test_probabilities = rf_model.predict_proba(
        X_test
    )[:, attack_class_index]

    if_test_scores = -if_model.score_samples(
        X_test
    )

    rf_test_pred = (
        rf_test_probabilities >= rf_threshold
    ).astype(int)

    if_test_pred = (
        if_test_scores >= if_threshold
    ).astype(int)

    hybrid_or_test = (
        (rf_test_pred == 1)
        | (if_test_pred == 1)
    ).astype(int)

    hybrid_and_test = (
        (rf_test_pred == 1)
        & (if_test_pred == 1)
    ).astype(int)

    test_candidates = {
        "random_forest": rf_test_pred,
        "isolation_forest": if_test_pred,
        "hybrid_or": hybrid_or_test,
        "hybrid_and": hybrid_and_test,
    }

    test_metrics = {}

    for name, predictions in test_candidates.items():

        metrics = calculate_metrics(
            y_test,
            predictions,
        )

        test_metrics[name] = metrics

        print(
            f"\nTEST {name}: "
            f"Accuracy={metrics['accuracy']:.4f}, "
            f"Precision={metrics['precision']:.4f}, "
            f"Recall={metrics['recall']:.4f}, "
            f"F1={metrics['f1_score']:.4f}, "
            f"FPR={metrics['false_positive_rate']:.4f}"
        )

        print(
            "Confusion matrix [benign, attack]:"
        )

        print(
            np.array(
                metrics["confusion_matrix"]
            )
        )

    # Save a row-by-row test report.
    results = pd.DataFrame({
        "actual_label": y_test.to_numpy(),
        "rf_attack_probability": rf_test_probabilities,
        "if_anomaly_score": if_test_scores,
        "rf_prediction": rf_test_pred,
        "if_prediction": if_test_pred,
        "hybrid_or_prediction": hybrid_or_test,
        "hybrid_and_prediction": hybrid_and_test,
    })

    results["selected_hybrid_prediction"] = (
        test_candidates[selected_strategy]
    )

    results["selected_hybrid_status"] = np.where(
        results["selected_hybrid_prediction"] == 1,
        "Potential Attack",
        "Benign",
    )

    results.to_csv(
        REPORT_DIR / "hybrid_test_predictions.csv",
        index=False,
    )

    # Save the metrics and experimental configuration.
    report = {
        "source_dataset": source_name,
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "features": feature_columns,
        "rf_threshold": rf_threshold,
        "if_threshold": if_threshold,
        "if_benign_quantile": IF_BENIGN_QUANTILE,
        "selected_strategy": selected_strategy,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }

    with open(
        REPORT_DIR / "hybrid_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    # Save both trained models and all required metadata.
    artifact = {
        "rf_model": rf_model,
        "if_model": if_model,
        "feature_columns": feature_columns,
        "feature_medians": feature_medians,
        "rf_threshold": rf_threshold,
        "if_threshold": if_threshold,
        "selected_strategy": selected_strategy,
        "source_dataset": source_name,
    }

    joblib.dump(
        artifact,
        MODEL_PATH,
    )

    print("\nTraining completed successfully.")

    print("Saved hybrid model:", MODEL_PATH)

    print(
        "Saved metrics:",
        REPORT_DIR / "hybrid_metrics.json",
    )

    print(
        "Saved predictions:",
        REPORT_DIR / "hybrid_test_predictions.csv",
    )


if __name__ == "__main__":
    main()