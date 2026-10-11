"""
Prepare UNSW-NB15 data for a separate network-level ML model.

The script:
1. Reads the official UNSW-NB15 training/testing CSV files.
2. Keeps attack_cat and label as target information.
3. Removes ID and target columns from the feature set.
4. Converts categorical network fields to numeric one-hot columns.
5. Handles missing/infinite values.
6. Ensures train/test have identical feature columns.
7. Saves processed datasets with the targets included.
"""

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "unsw"

TRAIN_FILE = DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_FILE = DATA_DIR / "UNSW_NB15_testing-set.csv"

OUTPUT_TRAIN = DATA_DIR / "processed_train.csv"
OUTPUT_TEST = DATA_DIR / "processed_test.csv"


def load_dataset(path):
    print(f"\nReading: {path.name}")

    df = pd.read_csv(path)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    return df


def clean_features(df):
    """
    Convert raw UNSW data into numeric model features.

    Returns only feature columns.
    """

    df = df.copy()

    # Clean column names.
    df.columns = [str(col).strip() for col in df.columns]

    # Keep target columns separately.
    target_columns = [
        "attack_cat",
        "label"
    ]

    for column in target_columns:
        if column not in df.columns:
            raise ValueError(
                f"Required target column '{column}' "
                f"was not found in the dataset."
            )

    # Remove columns that should not be model features.
    columns_to_drop = [
        "id",
        "attack_cat",
        "label"
    ]

    df = df.drop(
        columns=[
            column
            for column in columns_to_drop
            if column in df.columns
        ]
    )

    # Convert categorical network fields into
    # one-hot encoded numeric columns.
    categorical_columns = [
        column
        for column in ["proto", "service", "state"]
        if column in df.columns
    ]

    if categorical_columns:
        df = pd.get_dummies(
            df,
            columns=categorical_columns,
            dtype=int
        )

    # Replace infinity values.
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Fill missing numeric values with median.
    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    for column in numeric_columns:
        median = df[column].median()

        if pd.isna(median):
            median = 0

        df[column] = df[column].fillna(median)

    return df


def main():

    print("=" * 70)
    print("UNSW-NB15 DATA PREPARATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # Check input files
    # ---------------------------------------------------------

    if not TRAIN_FILE.exists():
        print("\nERROR:")
        print("Training file not found:")
        print(TRAIN_FILE)
        return

    if not TEST_FILE.exists():
        print("\nERROR:")
        print("Testing file not found:")
        print(TEST_FILE)
        return

    # ---------------------------------------------------------
    # Load raw datasets
    # ---------------------------------------------------------

    train_raw = load_dataset(TRAIN_FILE)
    test_raw = load_dataset(TEST_FILE)

    # ---------------------------------------------------------
    # Preserve targets
    # ---------------------------------------------------------

    train_attack_cat = train_raw["attack_cat"].fillna("Normal")
    train_label = train_raw["label"]

    test_attack_cat = test_raw["attack_cat"].fillna("Normal")
    test_label = test_raw["label"]

    # Remove accidental whitespace from attack names.
    train_attack_cat = train_attack_cat.astype(str).str.strip()
    test_attack_cat = test_attack_cat.astype(str).str.strip()

    # ---------------------------------------------------------
    # Clean feature data
    # ---------------------------------------------------------

    print("\nCleaning training data...")
    train_features = clean_features(train_raw)

    print("Cleaning testing data...")
    test_features = clean_features(test_raw)

    # ---------------------------------------------------------
    # Make train/test feature columns identical
    # ---------------------------------------------------------

    train_features, test_features = train_features.align(
        test_features,
        join="outer",
        axis=1,
        fill_value=0
    )

    # ---------------------------------------------------------
    # Add target columns back
    # ---------------------------------------------------------

    train_features["attack_cat"] = train_attack_cat.values
    train_features["label"] = train_label.values

    test_features["attack_cat"] = test_attack_cat.values
    test_features["label"] = test_label.values

    # ---------------------------------------------------------
    # Save processed datasets
    # ---------------------------------------------------------

    train_features.to_csv(
        OUTPUT_TRAIN,
        index=False
    )

    test_features.to_csv(
        OUTPUT_TEST,
        index=False
    )

    # ---------------------------------------------------------
    # Display results
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("PREPARATION COMPLETE")
    print("=" * 70)

    print("\nProcessed training file:")
    print(OUTPUT_TRAIN)

    print("\nProcessed testing file:")
    print(OUTPUT_TEST)

    print(f"\nTraining rows: {len(train_features)}")
    print(f"Testing rows:  {len(test_features)}")

    feature_count = len(
        train_features.columns
    ) - 2

    print(f"Feature columns: {feature_count}")

    print("\nTraining attack categories:")
    print(
        train_features["attack_cat"]
        .value_counts()
        .to_string()
    )

    print("\nTesting attack categories:")
    print(
        test_features["attack_cat"]
        .value_counts()
        .to_string()
    )

    print("\nTarget columns:")
    print("  attack_cat")
    print("  label")


if __name__ == "__main__":
    main()