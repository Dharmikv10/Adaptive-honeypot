"""Two-stage inference using the saved UNSW-NB15 XGBoost models.

This module is intentionally separate from the existing 19-feature
behavioral classifier. It does not train or modify either model.
"""

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent / "models"
BINARY_MODEL_PATH = MODEL_DIR / "unsw_binary_xgboost_clean.joblib"
ATTACK_MODEL_PATH = MODEL_DIR / "unsw_attack_xgboost_unweighted.joblib"


@lru_cache(maxsize=1)
def _load_models():
    """Load the two existing model artifacts once per process."""
    for path in (BINARY_MODEL_PATH, ATTACK_MODEL_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required model file not found: {path}")

    binary = joblib.load(BINARY_MODEL_PATH)
    attack = joblib.load(ATTACK_MODEL_PATH)

    binary_columns = binary["feature_columns"]
    attack_columns = attack["feature_columns"]

    if binary_columns != attack_columns:
        raise ValueError("Stage 1 and Stage 2 feature schemas do not match.")

    if len(binary_columns) != 194:
        raise ValueError(
            f"Expected 194 features, found {len(binary_columns)}."
        )

    if binary["model"].get_booster().num_features() != len(binary_columns):
        raise ValueError("Stage 1 model and feature metadata disagree.")

    if attack["model"].get_booster().num_features() != len(attack_columns):
        raise ValueError("Stage 2 model and feature metadata disagree.")

    return binary, attack


def predict_features(features):
    """Predict one row represented by a feature dictionary.

    Missing model columns are set to zero, and extra input keys are ignored.
    The caller must report which features were approximated or unavailable.
    """
    binary, attack = _load_models()
    columns = binary["feature_columns"]

    supplied_columns = set(features.keys()) & set(columns)
    missing_columns = [column for column in columns if column not in features]

    row = {column: features.get(column, 0) for column in columns}
    frame = pd.DataFrame([row], columns=columns, dtype=float)

    threshold = float(binary.get("threshold", 0.5))
    malicious_probability = float(binary["model"].predict_proba(frame)[0][1])
    is_malicious = malicious_probability >= threshold

    result = {
        "stage1_prediction": "Malicious" if is_malicious else "Benign",
        "stage1_malicious_probability": malicious_probability,
        "stage1_threshold": threshold,
        "stage2_prediction": None,
        "stage2_confidence": None,
        "stage2_probabilities": None,
        "feature_count": len(columns),
        "features_supplied": len(supplied_columns),
        "features_filled_with_zero": len(missing_columns),
        "missing_feature_names": missing_columns,
        "feature_schema_valid": True,
        "model_source": "UNSW-NB15 XGBoost",
    }

    # Stage 2 is only meaningful when Stage 1 marks the flow malicious.
    if is_malicious:
        probabilities = attack["model"].predict_proba(frame)[0]
        class_index = int(probabilities.argmax())
        encoder = attack.get("label_encoder")

        if encoder is not None:
            attack_name = str(encoder.inverse_transform([class_index])[0])
        else:
            classes = attack.get("classes", [])
            attack_name = (
                str(classes[class_index])
                if class_index < len(classes)
                else str(class_index)
            )

        result["stage2_prediction"] = attack_name
        result["stage2_confidence"] = float(probabilities[class_index])
        result["stage2_probabilities"] = {
            str(name): float(probability)
            for name, probability in zip(
                getattr(encoder, "classes_", attack.get("classes", [])),
                probabilities,
            )
        }

    return result


if __name__ == "__main__":
    binary, attack = _load_models()
    print("Both UNSW models loaded successfully.")
    print("Feature count:", len(binary["feature_columns"]))
    print("Stage 1 threshold:", binary.get("threshold", 0.5))
    print("Stage 2 classes:", list(attack["label_encoder"].classes_))
    print("This check loads models only; it does not run a traffic prediction.")
