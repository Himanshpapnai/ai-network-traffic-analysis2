from pathlib import Path

import joblib

from src.predict import MODEL_PATH


def test_model_file_exists():
    """Check that the training stage created the model."""

    assert MODEL_PATH.exists()


def test_model_artifact_structure():
    """Check that the saved artifact has the required keys."""

    artifact = joblib.load(MODEL_PATH)

    assert "model" in artifact
    assert "feature_columns" in artifact
    assert "feature_medians" in artifact

    assert len(artifact["feature_columns"]) > 0


def test_model_has_expected_feature_count():
    """Check that model metadata matches the feature list."""

    artifact = joblib.load(MODEL_PATH)

    assert (
        artifact["model"].n_features_in_
        == len(artifact["feature_columns"])
    )