from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "hybrid_model.joblib"


def load_artifact():
    """Load the saved hybrid model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Hybrid model not found. "
            "Run python -m src.train first."
        )

    return joblib.load(MODEL_PATH)


def prepare_prediction_features(df, artifact):
    """Align new input with the saved training features."""

    if df.empty:
        raise ValueError(
            "The input dataset is empty."
        )

    df = df.copy()

    df.columns = df.columns.str.strip()

    feature_columns = artifact["feature_columns"]

    medians = pd.Series(
        artifact["feature_medians"]
    ).reindex(feature_columns)

    # Check that the uploaded dataset resembles the training schema.
    available_columns = set(df.columns)

    matched_columns = [
        column
        for column in feature_columns
        if column in available_columns
    ]

    if len(matched_columns) < max(
        1,
        int(0.5 * len(feature_columns))
    ):
        raise ValueError(
            "Too few expected feature columns were found. "
            "Use a CSV with the same feature schema as the "
            "training dataset."
        )

    # Select features in precisely the training order.
    X = df.reindex(
        columns=feature_columns
    ).copy()

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Use training medians for missing values.
    X = X.fillna(medians)

    if X.isna().any().any():
        raise ValueError(
            "Some model features cannot be filled using "
            "the saved training medians."
        )

    if not np.isfinite(
        X.to_numpy(dtype=np.float64)
    ).all():
        raise ValueError(
            "Input contains invalid numerical values."
        )

    if (
        X.abs() > np.finfo(np.float32).max
    ).any().any():
        raise ValueError(
            "Input contains values outside the supported range."
        )

    return X


def predict_traffic(df):
    """
    Return predictions from Random Forest, Isolation Forest,
    and the selected hybrid decision strategy.

    Binary convention:
        0 = Benign
        1 = Potential Attack
    """

    artifact = load_artifact()

    X = prepare_prediction_features(
        df,
        artifact
    )

    rf_model = artifact["rf_model"]
    if_model = artifact["if_model"]

    # Random Forest attack probability.
    attack_class_index = list(
        rf_model.classes_
    ).index(1)

    rf_attack_probability = rf_model.predict_proba(
        X
    )[:, attack_class_index]

    rf_threshold = artifact["rf_threshold"]

    rf_prediction = (
        rf_attack_probability >= rf_threshold
    ).astype(int)

    # Isolation Forest anomaly score.
    # Higher scores indicate greater abnormality.
    anomaly_score = -if_model.score_samples(
        X
    )

    if_threshold = artifact["if_threshold"]

    if_prediction = (
        anomaly_score >= if_threshold
    ).astype(int)

    # Hybrid rules.
    hybrid_or = (
        (rf_prediction == 1)
        | (if_prediction == 1)
    ).astype(int)

    hybrid_and = (
        (rf_prediction == 1)
        & (if_prediction == 1)
    ).astype(int)

    strategy = artifact["selected_strategy"]

    if strategy == "random_forest":

        final_prediction = rf_prediction

    elif strategy == "hybrid_or":

        final_prediction = hybrid_or

    elif strategy == "hybrid_and":

        final_prediction = hybrid_and

    else:

        raise ValueError(
            f"Unknown hybrid strategy: {strategy}"
        )

    # Preserve original input columns for investigation.
    results = df.copy()

    results["rf_attack_probability"] = (
        rf_attack_probability
    )

    results["if_anomaly_score"] = anomaly_score

    results["anomaly_score"] = anomaly_score

    results["rf_prediction"] = rf_prediction

    results["if_prediction"] = if_prediction

    results["hybrid_or_prediction"] = hybrid_or

    results["hybrid_and_prediction"] = hybrid_and

    results["prediction"] = final_prediction

    results["status"] = np.where(
        final_prediction == 1,
        "Potential Anomaly",
        "Normal",
    )

    results["selected_strategy"] = strategy

    return results


if __name__ == "__main__":

    print("Hybrid prediction module is ready.")

    print("Model path:", MODEL_PATH)

    if MODEL_PATH.exists():

        artifact = load_artifact()

        print(
            "Selected strategy:",
            artifact["selected_strategy"]
        )

        print(
            "Expected feature count:",
            len(artifact["feature_columns"])
        )

    else:

        print(
            "Train the model first using: "
            "python -m src.train"
        )