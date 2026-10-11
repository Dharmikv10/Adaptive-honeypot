import os
import json
import joblib
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "unsw")
MODEL_DIR = os.path.join(BASE_DIR, "models")

TRAIN_FILE = os.path.join(DATA_DIR, "processed_train.csv")
TEST_FILE = os.path.join(DATA_DIR, "processed_test.csv")

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "unsw_binary_xgboost_clean.joblib"
)

METRICS_FILE = os.path.join(
    MODEL_DIR,
    "unsw_binary_xgboost_clean_metrics.json"
)

REPORT_FILE = os.path.join(
    MODEL_DIR,
    "unsw_binary_xgboost_clean_report.txt"
)

CM_FILE = os.path.join(
    MODEL_DIR,
    "unsw_binary_xgboost_clean_confusion_matrix.csv"
)

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 70)
print("UNSW-NB15 STAGE 1 - CLEAN BINARY XGBOOST")
print("=" * 70)

# ------------------------------------------------------------
# Load datasets
# ------------------------------------------------------------

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print("Training shape:", train_df.shape)
print("Testing shape :", test_df.shape)

# ------------------------------------------------------------
# Prepare features
# ------------------------------------------------------------

DROP_COLUMNS = [
    "attack_cat",
    "label",
    "stcpb",
    "dtcpb"
]

X = train_df.drop(
    columns=DROP_COLUMNS
)

y = train_df["label"].astype(int)

X_test = test_df.drop(
    columns=DROP_COLUMNS
)

y_test = test_df["label"].astype(int)

# Make sure test columns match training columns
X_test = X_test[X.columns]

print("\nDropped columns:")
for column in DROP_COLUMNS:
    print("  -", column)

print("\nRemaining features:", X.shape[1])
print("Training samples:", len(X))
print("Testing samples :", len(X_test))

print("\nTraining label distribution:")
print(y.value_counts())

print("\nTesting label distribution:")
print(y_test.value_counts())

# ------------------------------------------------------------
# Validation split
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("Creating validation split...")
print("=" * 70)

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining split  :", X_train.shape)
print("Validation split:", X_val.shape)

# ------------------------------------------------------------
# Class balance
# ------------------------------------------------------------

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count

print("\nClass balance:")
print("Benign samples   :", negative_count)
print("Malicious samples:", positive_count)
print("Scale pos weight :", scale_pos_weight)

# ------------------------------------------------------------
# Train XGBoost
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("Training CLEAN Stage 1 XGBoost...")
print("=" * 70)

model = XGBClassifier(
    objective="binary:logistic",

    n_estimators=400,
    max_depth=8,
    learning_rate=0.05,

    subsample=0.90,
    colsample_bytree=0.90,

    min_child_weight=1,

    gamma=0,

    reg_alpha=0.0,
    reg_lambda=1.0,

    scale_pos_weight=scale_pos_weight,

    tree_method="hist",
    n_jobs=2,

    random_state=42,

    eval_metric="logloss"
)

model.fit(
    X_train,
    y_train
)

print("\nTraining complete.")

# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)

val_pred = model.predict(X_val)
val_prob = model.predict_proba(X_val)[:, 1]

val_accuracy = accuracy_score(y_val, val_pred)
val_f1 = f1_score(y_val, val_pred)
val_auc = roc_auc_score(y_val, val_prob)

print(f"Validation Accuracy : {val_accuracy * 100:.2f}%")
print(f"Validation F1       : {val_f1:.4f}")
print(f"Validation ROC-AUC  : {val_auc:.4f}")

val_report = classification_report(
    y_val,
    val_pred,
    target_names=["Benign", "Malicious"],
    digits=4
)

print("\nValidation Classification Report:")
print(val_report)

# ------------------------------------------------------------
# Official test
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("OFFICIAL TEST RESULTS")
print("=" * 70)

test_pred = model.predict(X_test)
test_prob = model.predict_proba(X_test)[:, 1]

test_accuracy = accuracy_score(y_test, test_pred)
test_f1 = f1_score(y_test, test_pred)
test_auc = roc_auc_score(y_test, test_prob)

print(f"Test Accuracy : {test_accuracy * 100:.2f}%")
print(f"Test F1       : {test_f1:.4f}")
print(f"Test ROC-AUC  : {test_auc:.4f}")

test_report = classification_report(
    y_test,
    test_pred,
    target_names=["Benign", "Malicious"],
    digits=4
)

print("\nTest Classification Report:")
print(test_report)

# ------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------

cm = confusion_matrix(y_test, test_pred)

cm_df = pd.DataFrame(
    cm,
    index=["Actual_Benign", "Actual_Malicious"],
    columns=["Predicted_Benign", "Predicted_Malicious"]
)

print("\nConfusion Matrix:")
print(cm_df)

# ------------------------------------------------------------
# Save model
# ------------------------------------------------------------

print("\nSaving model...")

joblib.dump(
    {
        "model": model,
        "feature_columns": list(X.columns),
        "dropped_columns": DROP_COLUMNS,
        "threshold": 0.50,
        "stage": "binary_clean",
        "classes": {
            0: "Benign",
            1: "Malicious"
        }
    },
    MODEL_FILE,
    compress=3
)

# ------------------------------------------------------------
# Save metrics
# ------------------------------------------------------------

metrics = {
    "model": "UNSW Binary XGBoost Clean",
    "stage": 1,

    "dropped_columns": DROP_COLUMNS,
    "num_features": int(X.shape[1]),

    "n_estimators": 400,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.90,
    "colsample_bytree": 0.90,

    "validation_accuracy": float(val_accuracy),
    "validation_f1": float(val_f1),
    "validation_roc_auc": float(val_auc),

    "test_accuracy": float(test_accuracy),
    "test_f1": float(test_f1),
    "test_roc_auc": float(test_auc),

    "training_samples": int(len(X_train)),
    "validation_samples": int(len(X_val)),
    "testing_samples": int(len(X_test)),

    "scale_pos_weight": float(scale_pos_weight)
}

with open(METRICS_FILE, "w") as f:
    json.dump(metrics, f, indent=4)

with open(REPORT_FILE, "w") as f:
    f.write("VALIDATION RESULTS\n")
    f.write("==================\n")
    f.write(val_report)

    f.write("\n\nOFFICIAL TEST RESULTS\n")
    f.write("=====================\n")
    f.write(test_report)

cm_df.to_csv(CM_FILE)

print("\nFiles saved:")
print("  Model     :", MODEL_FILE)
print("  Metrics   :", METRICS_FILE)
print("  Report    :", REPORT_FILE)
print("  Confusion :", CM_FILE)

print("\n" + "=" * 70)
print("CLEAN STAGE 1 BINARY XGBOOST FINISHED")
print("=" * 70)