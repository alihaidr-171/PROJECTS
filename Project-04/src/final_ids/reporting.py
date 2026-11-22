
"""
Experiment Reporting Module

Saves CPU and GPU training results
into JSON and CSV files.
"""

import csv
import json
import platform

from datetime import datetime
from pathlib import Path

import torch

from .config import DOCS_DIR


def create_device_report(device_name, results):

    if results is None:
        return None

    return {
        "device": device_name,
        "training_time_seconds": results["train_seconds"],
        "inference_time_seconds": results["infer_seconds"],
        "throughput_flows_per_second": results["throughput"],
        "accuracy_percent": results["accuracy"],
        "balanced_accuracy_percent": results.get("balanced_accuracy"),
        "macro_precision_percent": results.get("macro_precision"),
        "macro_recall_percent": results.get("macro_recall"),
        "macro_f1_percent": results.get("macro_f1"),
        "weighted_f1_percent": results.get("weighted_f1"),
        "model_parameters": results["params"]
    }


def save_experiment_report(
    gpu_results,
    cpu_results,
    output_dir=DOCS_DIR
):

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    gpu_report = create_device_report(
        "GPU",
        gpu_results
    )

    cpu_report = create_device_report(
        "CPU",
        cpu_results
    )

    report = {
        "project": "GPU-Accelerated Deep Residual IDS",
        "dataset": "CICIDS2017",
        "date": datetime.now().astimezone().isoformat(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "Not Available"
        ),
        "results": {
            "gpu": gpu_report,
            "cpu": cpu_report
        }
    }

    # Calculate speedup when both results exist.
    if gpu_report is not None and cpu_report is not None:

        report["speedup"] = {
            "training": (
                cpu_report["training_time_seconds"]
                / max(gpu_report["training_time_seconds"], 1e-9)
            ),
            "inference": (
                cpu_report["inference_time_seconds"]
                / max(gpu_report["inference_time_seconds"], 1e-9)
            )
        }

    # Save JSON report.
    json_path = output_dir / "experiment_results.json"

    with open(json_path, "w", encoding="utf-8") as file:

        json.dump(
            report,
            file,
            indent=4
        )

    # Save CSV benchmark table.
    csv_path = output_dir / "benchmark_results.csv"

    rows = [
        result
        for result in [gpu_report, cpu_report]
        if result is not None
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as file:

        if rows:

            writer = csv.DictWriter(
                file,
                fieldnames=list(rows[0].keys())
            )

            writer.writeheader()
            writer.writerows(rows)

    print("\n[INFO] Experiment reports saved")
    print("JSON:", json_path)
    print("CSV:", csv_path)

    return json_path, csv_path


if __name__ == "__main__":

    print("Experiment Reporting Module")
    print("Output Directory:", DOCS_DIR)
