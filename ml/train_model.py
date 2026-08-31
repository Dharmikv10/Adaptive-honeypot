"""
ml/train_model.py

Trains both a Decision Tree and a Random Forest on the labeled session data
(synthetic for now -- see generate_training_data.py; swap in real Phase 6
Kali-attack sessions once you have them, same CSV schema). Evaluates both,
picks the Random Forest as the "active" model (usually more robust to noisy
features), but keeps both persisted so the dashboard/report can show the
side-by-side comparison your roadmap calls for.

Also persists per-class feature means -- used by predict.py's reason
extraction to explain *why* a session got its predicted label, by comparing
the session's own feature values against what's typical for that class.

Run:
    python3 ml/train_model.py
Reads:  ml/data/synthetic_sessions.csv
Writes: ml/models/random_forest.joblib
        ml/models/decision_tree.joblib
        ml/models/metrics.json
        ml/models/class_feature_means.json
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.append(str(Path(__file__).parent))
from feature_extraction import FEATURE_COLUMNS

DATA_PATH = Path(__file__).parent / "data" / "synthetic_sessions.csv"
MODELS_DIR = Path(__file__).parent / "models"
RANDOM_STATE = 42


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    dt = DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_pred)
    dt_report = classification_report(y_test, dt_pred, output_dict=True, zero_division=0)

    rf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_report = classification_report(y_test, rf_pred, output_dict=True, zero_division=0)

    print(f"Decision Tree accuracy:  {dt_accuracy:.4f}")
    print(f"Random Forest accuracy:  {rf_accuracy:.4f}")
    print("\n--- Random Forest classification report ---")
    print(classification_report(y_test, rf_pred, zero_division=0))

    labels_sorted = sorted(y.unique())
    dt_cm = confusion_matrix(y_test, dt_pred, labels=labels_sorted).tolist()
    rf_cm = confusion_matrix(y_test, rf_pred, labels=labels_sorted).tolist()

    joblib.dump(dt, MODELS_DIR / "decision_tree.joblib")
    joblib.dump(rf, MODELS_DIR / "random_forest.joblib")

    # Per-class feature means -- used later to explain *why* a prediction was
    # made (e.g. "auth_attempts here is close to the Brute Force average").
    class_means = df.groupby("label")[FEATURE_COLUMNS].mean().to_dict(orient="index")

    metrics = {
        "feature_columns": FEATURE_COLUMNS,
        "labels": labels_sorted,
        "active_model": "random_forest",
        "decision_tree": {
            "accuracy": dt_accuracy,
            "confusion_matrix": dt_cm,
            "feature_importances": dict(zip(FEATURE_COLUMNS, dt.feature_importances_.tolist())),
            "classification_report": dt_report,
        },
        "random_forest": {
            "accuracy": rf_accuracy,
            "confusion_matrix": rf_cm,
            "feature_importances": dict(zip(FEATURE_COLUMNS, rf.feature_importances_.tolist())),
            "classification_report": rf_report,
        },
        "training_samples": len(X_train),
        "test_samples": len(X_test),
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(MODELS_DIR / "class_feature_means.json", "w") as f:
        json.dump(class_means, f, indent=2)

    print(f"\nModels + metrics saved to {MODELS_DIR}/")


if __name__ == "__main__":
    main()