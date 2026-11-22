
"""
Project configuration for the CICIDS2017 intrusion detection system.
"""

from pathlib import Path

# Main project directory
PROJECT_DIR = Path(__file__).resolve().parents[2]

# Project folders
DATA_DIR = PROJECT_DIR / "data" / "cicids2017"
MODEL_DIR = PROJECT_DIR / "models"
DOCS_DIR = PROJECT_DIR / "docs"

# Dataset settings
CSV_CHUNK_SIZE = 100000
TEST_SIZE = 0.20

# Training settings
BATCH_SIZE = 512
EPOCHS = 12
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
SEED = 42

# Neural network settings
HIDDEN_WIDTH = 512
NUM_CLASSES = 7
DROPOUT = 0.20

# CICIDS2017 attack categories
CLASS_NAMES = [
    "BENIGN",
    "Botnet",
    "BruteForce",
    "DoS/DDoS",
    "Infiltration",
    "PortScan",
    "Web Attack"
]

# Convert class names to numerical indexes
CLASS_TO_IDX = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}

# Output file names
MODEL_FILENAME = "ids_cicids_deep_gpu.pth"
CHART_FILENAME = "cicids2017_benchmark.png"


if __name__ == "__main__":

    print("GPU-Accelerated IDS Configuration")

    print("\nProject Directory:", PROJECT_DIR)
    print("Dataset Directory:", DATA_DIR)
    print("Model Directory:", MODEL_DIR)
    print("Documentation Directory:", DOCS_DIR)

    print("\nTraining Settings")
    print("Batch Size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)
    print("Learning Rate:", LEARNING_RATE)
    print("Random Seed:", SEED)

    print("\nNetwork Settings")
    print("Hidden Width:", HIDDEN_WIDTH)
    print("Number of Classes:", NUM_CLASSES)

    print("\nAttack Categories:")

    for index, name in enumerate(CLASS_NAMES):
        print(index, "-", name)
