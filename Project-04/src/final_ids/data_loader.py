
"""
CICIDS2017 Dataset Loader

Responsibilities:
1. Find all CSV dataset files.
2. Read CSV files in chunks.
3. Convert attack labels into numerical values.
4. Clean invalid network traffic records.
5. Split training and testing data.
6. Normalize numerical features.
7. Convert data into PyTorch tensors.
"""

import gc
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import (
    DATA_DIR,
    CSV_CHUNK_SIZE,
    TEST_SIZE,
    SEED,
    CLASS_NAMES
)

from .labels import map_cicids_label


# Columns that are not numerical network features
DROP_COLUMNS = {
    "flow id",
    "source ip",
    "destination ip",
    "timestamp",
    "label"
}


# --------------------------------------------------
# 1. LOAD CSV FILES IN CHUNKS
# --------------------------------------------------

def load_csv_chunks(data_dir: Path):
    """
    Load CICIDS2017 CSV files in smaller chunks.
    This reduces temporary memory usage.
    """

    csv_files = sorted(data_dir.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found inside: {data_dir}"
        )

    print(f"[INFO] Found {len(csv_files)} CSV files")

    X_chunks = []
    y_chunks = []

    feature_columns = None

    for file_path in csv_files:

        print(f"\n[INFO] Reading: {file_path.name}")

        csv_reader = pd.read_csv(
            file_path,
            encoding="latin1",
            low_memory=True,
            chunksize=CSV_CHUNK_SIZE
        )

        for df in csv_reader:

            # Clean column names
            df.columns = (
                df.columns
                .str.strip()
                .str.replace("\ufeff", "", regex=False)
            )

            # Find label column
            label_columns = [
                col for col in df.columns
                if col.lower() == "label"
            ]

            if len(label_columns) != 1:
                raise ValueError(
                    f"Label column missing or duplicated: {file_path.name}"
                )

            label_column = label_columns[0]

            # Select numerical feature columns
            if feature_columns is None:

                feature_columns = [
                    col for col in df.columns
                    if col.lower() not in DROP_COLUMNS
                ]

                if not feature_columns:
                    raise ValueError(
                        "No numerical feature columns found."
                    )

            # Check feature consistency
            missing_columns = [
                col for col in feature_columns
                if col not in df.columns
            ]

            if missing_columns:
                raise ValueError(
                    f"Missing feature columns in {file_path.name}: "
                    f"{missing_columns[:5]}"
                )

            # Convert attack labels into class indexes
            y_chunk = df[label_column].map(
                map_cicids_label
            ).to_numpy(dtype=np.int64)

            # Convert features into numerical values
            X_chunk = (
                df[feature_columns]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(dtype=np.float32)
            )

            # Remove NaN and infinite values
            valid_mask = np.isfinite(X_chunk).all(axis=1)

            X_chunks.append(X_chunk[valid_mask])
            y_chunks.append(y_chunk[valid_mask])

            del df, X_chunk, y_chunk
            gc.collect()

    if not X_chunks:
        raise ValueError("No valid records were loaded.")

    # Combine all dataset chunks
    X_all = np.concatenate(X_chunks, axis=0)
    y_all = np.concatenate(y_chunks, axis=0)

    del X_chunks, y_chunks
    gc.collect()

    print("\n[INFO] CSV loading completed")
    print("Total records:", len(y_all))
    print("Total features:", X_all.shape[1])

    return X_all, y_all


# --------------------------------------------------
# 2. PREPARE TRAINING AND TESTING DATA
# --------------------------------------------------

def prepare_datasets(data_dir=DATA_DIR):
    """
    Load, split and scale the CICIDS2017 dataset.
    Return PyTorch tensors.
    """

    print("\n[INFO] Preparing CICIDS2017 dataset")

    X_all, y_all = load_csv_chunks(data_dir)

    # Display class distribution
    print("\n[INFO] Class Distribution")

    class_counts = np.bincount(
        y_all,
        minlength=len(CLASS_NAMES)
    )

    for index, class_name in enumerate(CLASS_NAMES):

        print(
            f"{class_name}: {class_counts[index]:,} records"
        )

    # Every class must have enough records for stratification
    if np.any(class_counts < 2):

        raise ValueError(
            "Every class must contain at least two records."
        )

    # Split data into 80% training and 20% testing
    print("\n[INFO] Splitting dataset")

    X_train, X_test, y_train, y_test = train_test_split(
        X_all,
        y_all,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y_all
    )

    del X_all, y_all
    gc.collect()

    # Fit scaler only on training data
    print("[INFO] Scaling features")

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    ).astype(np.float32, copy=False)

    X_test = scaler.transform(
        X_test
    ).astype(np.float32, copy=False)

    print("\n[INFO] Dataset preparation completed")

    print("Training shape:", X_train.shape)
    print("Testing shape:", X_test.shape)

    # Convert NumPy arrays into PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train)

    y_train_tensor = torch.from_numpy(
        y_train.astype(np.int64, copy=False)
    )

    X_test_tensor = torch.from_numpy(X_test)

    y_test_tensor = torch.from_numpy(
        y_test.astype(np.int64, copy=False)
    )

    return (
        X_train_tensor,
        y_train_tensor,
        X_test_tensor,
        y_test_tensor,
        scaler
    )


# --------------------------------------------------
# 3. TEST DATA LOADER
# --------------------------------------------------

if __name__ == "__main__":

    print("CICIDS2017 Data Loader Test")

    X_train, y_train, X_test, y_test, scaler = (
        prepare_datasets()
    )

    print("\nDataset Summary")

    print("Training Records:", len(X_train))
    print("Testing Records:", len(X_test))
    print("Number of Features:", X_train.shape[1])
    print("Number of Classes:", len(CLASS_NAMES))

    print("\nData Loader Working Successfully!")
