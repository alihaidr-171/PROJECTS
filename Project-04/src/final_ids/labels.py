
"""
CICIDS2017 Attack Label Mapping

Converts original dataset attack labels
into seven numerical classification categories.
"""

from .config import CLASS_TO_IDX, CLASS_NAMES


def map_cicids_label(raw_label):
    """Convert an attack label into a numerical class index."""

    label = str(raw_label).strip().lower()

    if label == "benign":
        return CLASS_TO_IDX["BENIGN"]

    elif "bot" in label:
        return CLASS_TO_IDX["Botnet"]

    elif "patator" in label:
        return CLASS_TO_IDX["BruteForce"]

    elif "dos" in label or "ddos" in label or "heartbleed" in label:
        return CLASS_TO_IDX["DoS/DDoS"]

    elif "portscan" in label:
        return CLASS_TO_IDX["PortScan"]

    elif "web attack" in label:
        return CLASS_TO_IDX["Web Attack"]

    elif "infiltration" in label:
        return CLASS_TO_IDX["Infiltration"]

    else:
        raise ValueError(
            f"Unknown CICIDS2017 label: {raw_label}"
        )


def get_class_name(class_index):
    """Convert a numerical index back into its class name."""

    if class_index < 0 or class_index >= len(CLASS_NAMES):
        raise ValueError("Invalid class index")

    return CLASS_NAMES[class_index]


if __name__ == "__main__":

    test_labels = [
        "BENIGN",
        "Bot",
        "FTP-Patator",
        "DoS Hulk",
        "Infiltration",
        "PortScan",
        "Web Attack - Brute Force"
    ]

    print("CICIDS2017 Label Mapping Test\n")

    for label in test_labels:

        index = map_cicids_label(label)

        name = get_class_name(index)

        print(f"{label:<30} -> {index} ({name})")
