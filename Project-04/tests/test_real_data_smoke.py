
"""
CICIDS2017 Real Data Pilot Test

Loads the actual dataset and tests a small sample
on both GPU and CPU.
"""

import torch

from src.final_ids.config import CLASS_NAMES
from src.final_ids.data_loader import prepare_datasets
from src.final_ids.trainer import train_and_benchmark


def sample_by_class(X, y, max_samples, seed=42):

    generator = torch.Generator()
    generator.manual_seed(seed)

    selected_indices = []

    for class_index in range(len(CLASS_NAMES)):

        class_indices = torch.where(
            y == class_index
        )[0]

        if len(class_indices) == 0:
            raise ValueError(
                f"No records found for {CLASS_NAMES[class_index]}"
            )

        order = torch.randperm(
            len(class_indices),
            generator=generator
        )

        chosen = class_indices[
            order[:max_samples]
        ]

        selected_indices.append(chosen)

    selected_indices = torch.cat(selected_indices)

    shuffle = torch.randperm(
        len(selected_indices),
        generator=generator
    )

    selected_indices = selected_indices[shuffle]

    return X[selected_indices], y[selected_indices]


def main():

    print("CICIDS2017 REAL DATA PILOT TEST")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    print("GPU:", torch.cuda.get_device_name(0))

    # Load the actual CICIDS2017 dataset.
    print("\nLoading real network traffic data...")

    X_train, y_train, X_test, y_test, scaler = (
        prepare_datasets()
    )

    # Select a small sample from every class.
    X_train_small, y_train_small = sample_by_class(
        X_train,
        y_train,
        max_samples=32
    )

    X_test_small, y_test_small = sample_by_class(
        X_test,
        y_test,
        max_samples=14
    )

    print("\nPilot Dataset")
    print("Training records:", len(X_train_small))
    print("Testing records:", len(X_test_small))
    print("Features:", X_train_small.shape[1])
    print("Classes:", len(CLASS_NAMES))

    # Test GPU training.
    gpu_results = train_and_benchmark(
        "cuda",
        X_train_small,
        y_train_small,
        X_test_small,
        y_test_small
    )

    print("\nGPU real-data pilot passed.")

    # Test CPU training.
    cpu_results = train_and_benchmark(
        "cpu",
        X_train_small,
        y_train_small,
        X_test_small,
        y_test_small
    )

    print("\nCPU real-data pilot passed.")

    # Check returned results.
    assert gpu_results["train_seconds"] > 0
    assert cpu_results["train_seconds"] > 0

    assert len(gpu_results["predictions"]) == len(y_test_small)
    assert len(cpu_results["predictions"]) == len(y_test_small)

    print("\nREAL DATA PILOT TEST PASSED!")


if __name__ == "__main__":
    main()
