
"""
Full CICIDS2017 One-Epoch CPU/GPU Benchmark

Uses the same real dataset, architecture and
training configuration on both devices.
"""

import gc
import torch

from src.final_ids.data_loader import prepare_datasets
from src.final_ids.trainer import train_and_benchmark


def main():

    print("FULL CICIDS2017 BENCHMARK")
    print("================================")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("GPU:", torch.cuda.get_device_name(0))

    # Load the complete dataset only once.
    X_train, y_train, X_test, y_test, scaler = (
        prepare_datasets()
    )

    print("\nDataset Information")
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))
    print("Features:", X_train.shape[1])

    # ---------------- GPU TEST ----------------

    print("\n========== GPU BENCHMARK ==========")

    torch.cuda.reset_peak_memory_stats()

    gpu_results = train_and_benchmark(
        "cuda",
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=1
    )

    gpu_training = gpu_results["train_seconds"]
    gpu_inference = gpu_results["infer_seconds"]
    gpu_accuracy = gpu_results["accuracy"]

    gpu_peak_memory = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
    )

    print(
        f"\nGPU Peak Memory: {gpu_peak_memory:.2f} MB"
    )

    # Release GPU model before CPU benchmark.
    del gpu_results

    gc.collect()
    torch.cuda.empty_cache()

    # ---------------- CPU TEST ----------------

    print("\n========== CPU BENCHMARK ==========")

    cpu_results = train_and_benchmark(
        "cpu",
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=1
    )

    cpu_training = cpu_results["train_seconds"]
    cpu_inference = cpu_results["infer_seconds"]
    cpu_accuracy = cpu_results["accuracy"]

    # ---------------- COMPARISON ----------------

    training_speedup = cpu_training / max(
        gpu_training, 1e-9
    )

    inference_speedup = cpu_inference / max(
        gpu_inference, 1e-9
    )

    print("\n================================")
    print("ONE-EPOCH CPU VS GPU COMPARISON")
    print("================================")

    print(f"GPU Training: {gpu_training:.2f} seconds")
    print(f"CPU Training: {cpu_training:.2f} seconds")

    print(f"GPU Accuracy: {gpu_accuracy:.2f}%")
    print(f"CPU Accuracy: {cpu_accuracy:.2f}%")

    print(f"GPU Inference: {gpu_inference:.4f} seconds")
    print(f"CPU Inference: {cpu_inference:.4f} seconds")

    print(f"GPU Peak VRAM: {gpu_peak_memory:.2f} MB")

    print(
        f"Training Speedup: {training_speedup:.2f}x"
    )

    print(
        f"Inference Speedup: {inference_speedup:.2f}x"
    )

    print("\nApproximate 12-epoch training time:")
    print(f"GPU: {gpu_training * 12:.2f} seconds")
    print(f"CPU: {cpu_training * 12:.2f} seconds")

    print("\nFULL DATASET BENCHMARK COMPLETED")

    # Release memory.
    del X_train, y_train, X_test, y_test
    del cpu_results, scaler

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
