
"""
Training and Benchmarking Module

Trains DeepResIDSNet on CPU and GPU,
then compares training and inference performance.
"""

import time
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

from .config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    SEED,
    CLASS_NAMES
)

from .model import DeepResIDSNet


# Wait for GPU operations to finish before timing.
def synchronize(device):

    if device.type == "cuda":
        torch.cuda.synchronize()


# Evaluate model predictions.
def evaluate_model(model, test_loader, y_test, device):

    model.eval()

    predictions = []

    synchronize(device)
    start = time.perf_counter()

    with torch.no_grad():

        for batch_x, batch_y in test_loader:

            batch_x = batch_x.to(
                device,
                non_blocking=(device.type == "cuda")
            )

            outputs = model(batch_x)

            predicted = torch.argmax(outputs, dim=1)

            predictions.append(predicted.cpu())

    synchronize(device)

    inference_time = time.perf_counter() - start

    y_pred = torch.cat(predictions)

    # Overall accuracy
    accuracy = (
        (y_pred == y_test).float().mean().item()
    ) * 100

    throughput = len(y_test) / max(inference_time, 1e-6)

    print("\nEvaluation Results")

    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Inference Time: {inference_time:.4f} seconds")
    print(f"Throughput: {throughput:,.0f} flows/second")

    # Class-wise accuracy
    print("\nClass-wise Accuracy")

    for index, class_name in enumerate(CLASS_NAMES):

        mask = y_test == index
        count = int(mask.sum().item())

        if count > 0:

            class_accuracy = (
                (y_pred[mask] == y_test[mask])
                .float()
                .mean()
                .item()
            ) * 100

            print(
                f"{class_name:<15} "
                f"{class_accuracy:.2f}% "
                f"({count:,} records)"
            )

    return {
        "predictions": y_pred,
        "accuracy": accuracy,
        "inference_time": inference_time,
        "throughput": throughput
    }


# Train and benchmark the model on one device.
def train_and_benchmark(
    device_name,
    X_train,
    y_train,
    X_test,
    y_test
):

    torch.manual_seed(SEED)

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device(device_name)

    input_dim = X_train.shape[1]

    model = DeepResIDSNet(
        input_dim=input_dim,
        num_classes=len(CLASS_NAMES)
    ).to(device)

    parameter_count = sum(
        p.numel() for p in model.parameters()
    )

    print("\n===================================")
    print("Training on:", device_name.upper())

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("Model Parameters:", f"{parameter_count:,}")
    print("Batch Size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)
    print("===================================")

    # Prepare training and testing datasets.
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=(len(train_dataset) > BATCH_SIZE),
        generator=torch.Generator().manual_seed(SEED)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # Define loss function and optimizer.
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0001
    )

    # GPU warm-up
    if device.type == "cuda":

        model.eval()

        with torch.no_grad():

            sample = X_train[:BATCH_SIZE].to(device)
            model(sample)
            del sample

        synchronize(device)

    # Training loop
    model.train()

    synchronize(device)
    start_time = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):

        epoch_start = time.perf_counter()

        total_loss = 0.0
        total_samples = 0

        model.train()

        for batch_x, batch_y in train_loader:

            batch_x = batch_x.to(
                device,
                non_blocking=(device.type == "cuda")
            )

            batch_y = batch_y.to(
                device,
                non_blocking=(device.type == "cuda")
            )

            optimizer.zero_grad()

            outputs = model(batch_x)

            loss = criterion(outputs, batch_y)

            loss.backward()

            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            total_samples += batch_x.size(0)

        synchronize(device)

        epoch_time = time.perf_counter() - epoch_start

        average_loss = total_loss / max(total_samples, 1)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Loss: {average_loss:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )

    synchronize(device)

    training_time = time.perf_counter() - start_time

    print(f"\nTotal Training Time: {training_time:.2f} seconds")

    # Evaluate trained model.
    results = evaluate_model(
        model,
        test_loader,
        y_test,
        device
    )

    print("\nTraining and evaluation completed.")

    return {
        "model": model,
        "params": parameter_count,
        "train_seconds": training_time,
        "infer_seconds": results["inference_time"],
        "throughput": results["throughput"],
        "accuracy": results["accuracy"],
        "predictions": results["predictions"]
    }
