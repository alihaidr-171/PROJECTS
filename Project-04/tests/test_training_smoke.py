
"""
CPU and GPU Training Smoke Test

Checks that the training pipeline works
on both CPU and CUDA using synthetic data.
"""

import torch

from src.final_ids.trainer import train_and_benchmark


def main():

    print("CPU AND GPU TRAINING SMOKE TEST")

    if torch.cuda.is_available():
        print("Detected GPU:", torch.cuda.get_device_name(0))
    else:
        print("CUDA unavailable. CPU test only.")

    # Create small synthetic training data.
    torch.manual_seed(42)

    X_train = torch.randn(224, 78)
    y_train = torch.arange(7).repeat_interleave(32)

    X_test = torch.randn(56, 78)
    y_test = torch.arange(7).repeat_interleave(8)

    print("\nDataset Information")
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))
    print("Features:", X_train.shape[1])
    print("Classes:", len(torch.unique(y_train)))

    devices = []

    if torch.cuda.is_available():
        devices.append("cuda")

    devices.append("cpu")

    for device_name in devices:

        print("\nTesting:", device_name.upper())

        results = train_and_benchmark(
            device_name,
            X_train,
            y_train,
            X_test,
            y_test
        )

        # Verify required result fields.
        required_keys = [
            "model",
            "params",
            "train_seconds",
            "infer_seconds",
            "throughput",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "predictions",
            "true_labels"
        ]

        for key in required_keys:

            assert key in results, (
                f"Missing result key: {key}"
            )

        # Verify prediction and label counts.
        assert len(results["predictions"]) == len(y_test)
        assert len(results["true_labels"]) == len(y_test)

        # Verify training and inference timings.
        assert results["train_seconds"] > 0
        assert results["infer_seconds"] > 0

        print(device_name.upper(), "TEST PASSED")

    print("\nCPU AND GPU TRAINING TESTS PASSED!")


if __name__ == "__main__":
    main()
