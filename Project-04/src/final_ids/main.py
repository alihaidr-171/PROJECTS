
"""
Main Application
GPU-Accelerated Deep Residual Intrusion Detection System
Dataset: CICIDS2017
"""

import gc
import numpy as np
import torch

from .config import (
    DATA_DIR,
    MODEL_DIR,
    MODEL_FILENAME,
    CLASS_NAMES,
    SEED
)

from .data_loader import prepare_datasets

from .trainer import train_and_benchmark

from .visualization import save_benchmark_chart


# --------------------------------------------------
# 1. CHECK PROJECT ENVIRONMENT
# --------------------------------------------------

def check_environment():

    print("GPU-Accelerated Intrusion Detection System")
    print("Dataset: CICIDS2017")

    print("\nChecking environment...")

    print("PyTorch:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "GPU Memory:",
            round(
                torch.cuda.get_device_properties(0).total_memory
                / (1024 ** 3),
                2
            ),
            "GB"
        )

    else:

        print("GPU not available. CPU mode will be used.")

    if not DATA_DIR.exists():

        raise FileNotFoundError(
            f"Dataset directory not found: {DATA_DIR}"
        )

    csv_files = list(DATA_DIR.rglob("*.csv"))

    if not csv_files:

        raise FileNotFoundError(
            f"No CICIDS2017 CSV files found inside: {DATA_DIR}"
        )

    print("Dataset CSV files found:", len(csv_files))


# --------------------------------------------------
# 2. SAVE TRAINED MODEL
# --------------------------------------------------

def save_model(results, input_dim, scaler, device_name):

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if device_name == "cuda":
        filename = MODEL_FILENAME
    else:
        filename = "ids_cicids_deep_cpu.pth"

    model_path = MODEL_DIR / filename

    checkpoint = {

        "model_state_dict": results["model"].state_dict(),

        "input_dim": input_dim,

        "num_classes": len(CLASS_NAMES),

        "class_names": CLASS_NAMES,

        "scaler_mean": scaler.mean_,

        "scaler_scale": scaler.scale_,

        "device": device_name
    }

    torch.save(
        checkpoint,
        model_path
    )

    print("\n[INFO] Model saved successfully:")
    print(model_path)


# --------------------------------------------------
# 3. CPU AND GPU PERFORMANCE COMPARISON
# --------------------------------------------------

def show_comparison(gpu_results, cpu_results):

    print("\n========================================")
    print("CICIDS2017 FINAL BENCHMARK")
    print("========================================")

    print("\nGPU Training Time:",
          round(gpu_results["train_seconds"], 2), "seconds")

    print("CPU Training Time:",
          round(cpu_results["train_seconds"], 2), "seconds")

    print("\nGPU Accuracy:",
          round(gpu_results["accuracy"], 2), "%")

    print("CPU Accuracy:",
          round(cpu_results["accuracy"], 2), "%")

    print("\nGPU Inference Time:",
          round(gpu_results["infer_seconds"], 4), "seconds")

    print("CPU Inference Time:",
          round(cpu_results["infer_seconds"], 4), "seconds")

    train_speedup = (
        cpu_results["train_seconds"]
        / max(gpu_results["train_seconds"], 1e-9)
    )

    inference_speedup = (
        cpu_results["infer_seconds"]
        / max(gpu_results["infer_seconds"], 1e-9)
    )

    print("\nGPU Training Speedup:",
          round(train_speedup, 2), "x")

    print("GPU Inference Speedup:",
          round(inference_speedup, 2), "x")

    print("\n========================================")


# --------------------------------------------------
# 4. MAIN FUNCTION
# --------------------------------------------------

def main():

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Check environment and dataset.
    check_environment()

    # Load and preprocess dataset.
    print("\nPreparing CICIDS2017 dataset...")

    (
        X_train,
        y_train,
        X_test,
        y_test,
        scaler
    ) = prepare_datasets(DATA_DIR)

    print("\nDataset preparation completed.")

    input_dim = X_train.shape[1]

    print("Input Features:", input_dim)
    print("Classification Categories:", len(CLASS_NAMES))

    # Train using GPU when available.
    gpu_results = None

    if torch.cuda.is_available():

        gpu_results = train_and_benchmark(
            "cuda",
            X_train,
            y_train,
            X_test,
            y_test
        )

        save_model(
            gpu_results,
            input_dim,
            scaler,
            "cuda"
        )

    # Train using CPU.
    cpu_results = train_and_benchmark(
        "cpu",
        X_train,
        y_train,
        X_test,
        y_test
    )

    if gpu_results is not None:

        # Display comparison.
        show_comparison(
            gpu_results,
            cpu_results
        )

        # Generate performance chart.
        save_benchmark_chart(
            gpu_results,
            cpu_results
        )

    else:

        print("\nOnly CPU results are available.")

        save_model(
            cpu_results,
            input_dim,
            scaler,
            "cpu"
        )

    # Release dataset memory.
    del X_train, y_train
    del X_test, y_test

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nIDS APPLICATION COMPLETED SUCCESSFULLY!")


# --------------------------------------------------
# 5. APPLICATION ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    main()
