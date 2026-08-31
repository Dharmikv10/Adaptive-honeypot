"""
ml/predict.py

Loads the trained Random Forest (the "active" model per metrics.json) and
predicts the attack type for a given session_id, with:
  - a confidence score (max class probability)
  - a plain-language "reason" explaining the prediction, built from two
    signals combined: (1) which features the model weighs most heavily in
    general (global feature_importances_), and (2) how this session's own
    values on those features compare to the per-class averages seen during
    training. This is a lightweight, transparent alternative to a full
    SHAP/LIME explainer -- accurate enough to narrate live in a demo, and
    it's honest about being importance-based rather than a per-instance
    causal explanation.

Usage (standalone test):
    python3 ml/predict.py <session_id>

Usage (from other code, e.g. the adaptation engine in Phase 4):
    from ml.predict import predict_session
    result = predict_session(session_id)
    # result = {"predicted_class": ..., "confidence": ..., "reason": ...,
    #           "features": {...}, "all_probabilities": {...}}
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.append(str(Path(__file__).parent))
from feature_extraction import extract_features, features_to_vector, FEATURE_COLUMNS

MODELS_DIR = Path(__file__).parent / "models"

_model = None
_metrics = None
_class_means = None


def _load():
    """Lazy-load the model + metadata once, reused across calls."""
    global _model, _metrics, _class_means
    if _model is None:
        with open(MODELS_DIR / "metrics.json") as f:
            _metrics = json.load(f)
        active = _metrics["active_model"]  # "random_forest"
        model_file = "random_forest.joblib" if active == "random_forest" else "decision_tree.joblib"
        _model = joblib.load(MODELS_DIR / model_file)
        with open(MODELS_DIR / "class_feature_means.json") as f:
            _class_means = json.load(f)
    return _model, _metrics, _class_means


_PHRASES = {
    "auth_attempts": lambda v: f"{int(v)} login attempts",
    "unique_usernames": lambda v: f"{int(v)} distinct usernames",
    "unique_passwords": lambda v: f"{int(v)} distinct passwords",
    "command_count": lambda v: f"{int(v)} commands/requests",
    "duration_seconds": lambda v: f"a session lasting {v:.1f}s",
    "avg_seconds_between": lambda v: f"only {v:.2f}s between actions",
    "recent_session_count_from_ip": lambda v: f"{int(v)} recent connections from this IP",
    "username_entropy": lambda v: f"username randomness of {v:.1f} bits",
    "password_entropy": lambda v: f"password randomness of {v:.1f} bits",
    "repeated_password_ratio": lambda v: f"{v*100:.0f}% repeated passwords",
    "unique_payloads": lambda v: f"{int(v)} distinct payloads/paths",
    "avg_payload_length": lambda v: f"an average payload length of {v:.0f} characters",
    "commands_per_minute": lambda v: f"a request rate of {v:.0f}/minute",
    "event_count": lambda v: f"{int(v)} total events",
}


def _build_reason(features: dict, predicted_class: str, confidence: float,
                   metrics: dict, class_means: dict) -> str:
    """
    Builds a narrative explanation like:
    "Classified as Brute Force with 94% confidence: the session showed 27
    login attempts and only 0.31s between actions, both far above what's
    typical for other classes -- consistent with an automated password-
    guessing tool rather than a human or a different attack pattern."
    """
    importances = metrics["random_forest"]["feature_importances"]
    top_features = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:3]

    class_avg = class_means.get(predicted_class, {})
    phrases = []
    for feat_name, _importance in top_features:
        session_val = features.get(feat_name)
        class_typical = class_avg.get(feat_name)
        if session_val is None or class_typical is None:
            continue
        phrase_fn = _PHRASES.get(feat_name)
        phrase = phrase_fn(session_val) if phrase_fn else f"{feat_name}={session_val}"
        direction = "in line with" if abs(session_val - class_typical) <= max(1.0, class_typical * 0.5) \
            else ("well above" if session_val > class_typical else "well below")
        phrases.append(f"{phrase} ({direction} typical {predicted_class} behavior)")

    confidence_word = "high" if confidence >= 0.85 else ("moderate" if confidence >= 0.6 else "low")

    if not phrases:
        return (
            f"Classified as {predicted_class} with {confidence*100:.0f}% ({confidence_word}) "
            f"confidence, based on overall session behavior."
        )

    return (
        f"Classified as {predicted_class} with {confidence*100:.0f}% ({confidence_word}) confidence. "
        f"The session showed " + "; ".join(phrases) + "."
    )


def predict_session(session_id: str) -> dict:
    model, metrics, class_means = _load()

    features = extract_features(session_id)
    vector = pd.DataFrame([features_to_vector(features)], columns=FEATURE_COLUMNS)

    predicted_class = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]
    class_labels = model.classes_

    prob_map = {label: float(p) for label, p in zip(class_labels, probabilities)}
    confidence = prob_map[predicted_class]

    reason = _build_reason(features, predicted_class, confidence, metrics, class_means)

    return {
        "session_id": session_id,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "all_probabilities": {k: round(v, 4) for k, v in prob_map.items()},
        "reason": reason,
        "features": {k: features[k] for k in FEATURE_COLUMNS},
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 ml/predict.py <session_id>")
        sys.exit(1)
    result = predict_session(sys.argv[1])
    print(json.dumps(result, indent=2))