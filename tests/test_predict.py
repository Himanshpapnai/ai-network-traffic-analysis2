
import joblib

from src.predict import MODEL_PATH


def test_model_file_exists():
    assert MODEL_PATH.exists()


def test_model_artifact_structure():
    artifact = joblib.load(MODEL_PATH)

    required_keys = {
        "rf_model",
        "if_model",
        "feature_columns",
        "feature_medians",
        "rf_threshold",
        "if_threshold",
        "selected_strategy",
    }

    assert required_keys.issubset(artifact.keys())
    assert len(artifact["feature_columns"]) > 0


def test_model_has_expected_feature_count():
    artifact = joblib.load(MODEL_PATH)

    feature_count = len(artifact["feature_columns"])

    assert artifact["rf_model"].n_features_in_ == feature_count
    assert artifact["if_model"].n_features_in_ == feature_count
  
