import os
import json
import joblib
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "unsw")
MODEL_DIR = os.path.join(BASE_DIR, "models")

TRAIN_FILE = os.path.join(DATA_DIR, "processed_train.csv")
TEST_FILE = os.path.join(DATA_DIR, "processed_test.csv")

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "unsw_attack_xgboost_unweighted.joblib"
)

METRICS_FILE = os.path.join(
    MODEL_DIR,
    "unsw_attack_xgboost_unweighted_metrics.json"
)

REPORT_FILE = os.path.join(
    MODEL_DIR,
    "unsw_attack_xgboost_unweighted_report.txt"
)

CM_FILE = os.path.join(
    MODEL_DIR,
    "unsw_attack_xgboost_unweighted_confusion_matrix.csv"
)

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 70)
print("UNSW-NB15 STAGE 2 - MALICIOUS ATTACK CLASSIFICATION")
print("=" * 70)

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print("Original training shape:", train_df.shape)
print("Original testing shape :", test_df.shape)

# ------------------------------------------------------------
# Keep malicious traffic only
# ------------------------------------------------------------

train_df = train_df[
    (train_df["label"] == 1) &
    (train_df["attack_cat"].str.lower() != "normal")
].copy()

test_df = test_df[
    (test_df["label"] == 1) &
    (test_df["attack_cat"].str.lower() != "normal")
].copy()

print("\nAfter keeping malicious traffic:")
print("Training shape:", train_df.shape)
print("Testing shape :", test_df.shape)

print("\nTraining attack distribution:")
print(train_df["attack_cat"].value_counts())

print("\nTesting attack distribution:")
print(test_df["attack_cat"].value_counts())

# ------------------------------------------------------------
# Feature preparation
# ------------------------------------------------------------

DROP_COLUMNS = [
    "attack_cat",
    "label",
    "stcpb",
    "dtcpb"
]

X = train_df.drop(columns=DROP_COLUMNS)
X_test = test_df.drop(columns=DROP_COLUMNS)

# Make test columns identical to training
X_test = X_test[X.columns]

# ------------------------------------------------------------
# Encode attack categories
# ------------------------------------------------------------

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(
    train_df["attack_cat"]
)

y_test = label_encoder.transform(
    test_df["attack_cat"]
)

class_names = list(label_encoder.classes_)

print("\nAttack classes:")
for index, name in enumerate(class_names):
    print(f"  {index}: {name}")

print("\nNumber of attack classes:", len(class_names))
print("Features:", X.shape[1])

# ------------------------------------------------------------
# Train / validation split
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
# Calculate class weights
# ------------------------------------------------------------

print("\nCalculating class weights...")

class_counts = pd.Series(y_train).value_counts().sort_index()

total_samples = len(y_train)
num_classes = len(class_names)

class_weights = {}

for class_id in range(num_classes):
    count = class_counts[class_id]

    weight = total_samples / (
        num_classes * count
    )

    class_weights[class_id] = weight

    print(
        f"{class_names[class_id]:15s} "
        f"count={count:6d} "
        f"weight={weight:.4f}"
    )

sample_weights = pd.Series(y_train).map(
    class_weights
).values

# ------------------------------------------------------------
# Train XGBoost
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("Training Stage 2 XGBoost...")
print("=" * 70)

model = XGBClassifier(
    objective="multi:softprob",

    num_class=num_classes,

    n_estimators=400,
    max_depth=8,
    learning_rate=0.05,

    subsample=0.90,
    colsample_bytree=0.90,

    min_child_weight=1,

    gamma=0,

    reg_alpha=0.0,
    reg_lambda=1.0,

    tree_method="hist",
    n_jobs=2,

    random_state=42,

    eval_metric="mlogloss"
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

val_accuracy = accuracy_score(
    y_val,
    val_pred
)

val_macro_f1 = f1_score(
    y_val,
    val_pred,
    average="macro"
)

val_weighted_f1 = f1_score(
    y_val,
    val_pred,
    average="weighted"
)

print(f"Validation Accuracy   : {val_accuracy * 100:.2f}%")
print(f"Validation Macro F1   : {val_macro_f1:.4f}")
print(f"Validation Weighted F1: {val_weighted_f1:.4f}")

val_report = classification_report(
    y_val,
    val_pred,
    target_names=class_names,
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

test_accuracy = accuracy_score(
    y_test,
    test_pred
)

test_macro_f1 = f1_score(
    y_test,
    test_pred,
    average="macro"
)

test_weighted_f1 = f1_score(
    y_test,
    test_pred,
    average="weighted"
)

print(f"Test Accuracy   : {test_accuracy * 100:.2f}%")
print(f"Test Macro F1   : {test_macro_f1:.4f}")
print(f"Test Weighted F1: {test_weighted_f1:.4f}")

test_report = classification_report(
    y_test,
    test_pred,
    target_names=class_names,
    digits=4
)

print("\nTest Classification Report:")
print(test_report)

# ------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    test_pred
)

cm_df = pd.DataFrame(
    cm,
    index=[
        f"Actual_{name}"
        for name in class_names
    ],
    columns=[
        f"Predicted_{name}"
        for name in class_names
    ]
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
        "label_encoder": label_encoder,
        "feature_columns": list(X.columns),
        "dropped_columns": DROP_COLUMNS,
        "stage": "attack_classification",
        "classes": class_names
    },
    MODEL_FILE,
    compress=3
)

# ------------------------------------------------------------
# Save metrics
# ------------------------------------------------------------

metrics = {
    "model": "UNSW Stage 2 Attack XGBoost",

    "stage": 2,

    "num_classes": num_classes,
    "classes": class_names,

    "dropped_columns": DROP_COLUMNS,
    "num_features": int(X.shape[1]),

    "n_estimators": 400,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.90,
    "colsample_bytree": 0.90,

    "validation_accuracy": float(val_accuracy),
    "validation_macro_f1": float(val_macro_f1),
    "validation_weighted_f1": float(val_weighted_f1),

    "test_accuracy": float(test_accuracy),
    "test_macro_f1": float(test_macro_f1),
    "test_weighted_f1": float(test_weighted_f1),

    "training_samples": int(len(X_train)),
    "validation_samples": int(len(X_val)),
    "testing_samples": int(len(X_test))
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
print("STAGE 2 ATTACK CLASSIFICATION FINISHED")
print("=" * 70)