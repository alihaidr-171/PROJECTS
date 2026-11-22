
"""
Visualization Module for CICIDS2017 IDS.

Creates CPU vs GPU performance comparison charts.
"""

import matplotlib.pyplot as plt

from pathlib import Path

from .config import DOCS_DIR, CHART_FILENAME


def save_benchmark_chart(
    gpu_results,
    cpu_results,
    output_path=None
):
    """
    Generate comparison charts for:
    1. Training time
    2. Inference time
    3. Inference throughput
    4. Test accuracy
    """

    if gpu_results is None or cpu_results is None:
        raise ValueError(
            "Both GPU and CPU results are required."
        )

    # Default chart location
    if output_path is None:
        output_path = DOCS_DIR / CHART_FILENAME

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Collect performance measurements
    metrics = {
        "Training Time (Seconds)": (
            gpu_results["train_seconds"],
            cpu_results["train_seconds"]
        ),

        "Inference Time (Seconds)": (
            gpu_results["infer_seconds"],
            cpu_results["infer_seconds"]
        ),

        "Inference Throughput (Flows/Second)": (
            gpu_results["throughput"],
            cpu_results["throughput"]
        ),

        "Test Accuracy (%)": (
            gpu_results["accuracy"],
            cpu_results["accuracy"]
        )
    }

    # Create four comparison charts
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8)
    )

    fig.suptitle(
        "CICIDS2017 Deep Residual IDS - CPU vs GPU",
        fontsize=14,
        fontweight="bold"
    )

    axes = axes.flatten()

    for index, (title, values) in enumerate(metrics.items()):

        gpu_value, cpu_value = values

        bars = axes[index].bar(
            ["GPU", "CPU"],
            [gpu_value, cpu_value]
        )

        axes[index].set_title(title)

        axes[index].set_ylabel(title)

        # Display values above each bar
        for bar in bars:

            height = bar.get_height()

            axes[index].text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom"
            )

        axes[index].grid(
            axis="y",
            linestyle="--",
            alpha=0.3
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    # Calculate speedup
    training_speedup = (
        cpu_results["train_seconds"]
        / max(gpu_results["train_seconds"], 1e-9)
    )

    inference_speedup = (
        cpu_results["infer_seconds"]
        / max(gpu_results["infer_seconds"], 1e-9)
    )

    print("\n[INFO] Performance chart generated")

    print("Chart location:", output_path)

    print(
        f"GPU Training Speedup: "
        f"{training_speedup:.2f}x"
    )

    print(
        f"GPU Inference Speedup: "
        f"{inference_speedup:.2f}x"
    )

    return output_path


if __name__ == "__main__":

    print("CICIDS2017 Visualization Module")
    print("Output Directory:", DOCS_DIR)
    print("Chart Filename:", CHART_FILENAME)
