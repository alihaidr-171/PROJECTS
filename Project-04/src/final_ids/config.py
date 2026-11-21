
from pathlib import Path

# Project directory
PROJECT_DIR = Path(__file__).resolve().parents[2]

# Dataset directory
DATA_DIR = PROJECT_DIR / "data" / "cicids2017"

# Model storage directory
MODEL_DIR = PROJECT_DIR / "models"

# Documentation directory
DOCS_DIR = PROJECT_DIR / "docs"

# Training settings
BATCH_SIZE = 4096
EPOCHS = 12
LEARNING_RATE = 0.001
SEED = 42

# Seven intrusion detection classes
CLASS_NAMES = [
    "BENIGN",
    "Botnet",
    "BruteForce",
    "DoS/DDoS",
    "Infiltration",
    "PortScan",
    "Web Attack"
]

# Convert class names into numerical labels
CLASS_TO_IDX = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}

# Display project configuration
if __name__ == "__main__":

    print("Project Directory:", PROJECT_DIR)
    print("Dataset Directory:", DATA_DIR)
    print("Model Directory:", MODEL_DIR)
    print("Documentation Directory:", DOCS_DIR)

    print("\nTraining Configuration")
    print("Batch Size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)
    print("Learning Rate:", LEARNING_RATE)

    print("\nNumber of Classes:", len(CLASS_NAMES))
    print("Classes:", CLASS_NAMES)
