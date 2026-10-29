from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "isolation_forest.joblib"


def load_artifact():
    """Load the trained model and its feature metadata."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run python src/train.py first."
        )

    return joblib.load(MODEL_PATH)


def prepare_prediction_features(df, artifact):
    """
    Align new traffic data with the model's training features.
    """

    df = df.copy()

    df.columns = df.columns.str.strip()

    # Remove labels and common flow identifiers.
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

    df = df.drop(
        columns=[
            col for col in df.columns
            if col.lower() in excluded_columns
        ],
        errors="ignore",
    )

    feature_columns = artifact["feature_columns"]
    medians = artifact["feature_medians"]

    # Reindex to guarantee the exact training feature order.
    X = df.reindex(columns=feature_columns)

    # Convert numeric features and handle invalid values.
    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(
        pd.Series(medians)
        .reindex(feature_columns)
    )

    # Do not silently accept entirely unusable input rows.
    if X.isna().any().any():
        raise ValueError(
            "Some required model features have no valid values "
            "or training medians."
        )

    if not np.isfinite(X.to_numpy(dtype=np.float64)).all():
        raise ValueError(
            "Input contains invalid numerical values."
        )

    if (X.abs() > np.finfo(np.float32).max).any().any():
        raise ValueError(
            "Input contains values outside the supported range."
        )

    return X


def predict_traffic(df):
    """
    Analyse a DataFrame of network flows.

    Prediction:
        1  = Normal
       -1  = Potential anomaly

    The anomaly score is higher for more unusual observations
    when expressed as the negative of score_samples().
    """

    if df.empty:
        raise ValueError("The input dataset is empty.")

    artifact = load_artifact()

    X = prepare_prediction_features(df, artifact)

    model = artifact["model"]

    predictions = model.predict(X)

    # Negating score_samples gives an intuitive direction:
    # larger values indicate greater abnormality.
    anomaly_scores = -model.score_samples(X)

    results = df.copy()

    results["anomaly_score"] = anomaly_scores

    results["prediction"] = predictions

    results["status"] = np.where(
        predictions == -1,
        "Potential Anomaly",
        "Normal",
    )

    return results


if __name__ == "__main__":
    print("Prediction module is ready.")
    print("Model path:", MODEL_PATH)