
"""
Deep Residual Neural Network for CICIDS2017.

This file contains:
1. ResidualBlock
2. DeepResIDSNet
3. Model testing
"""

import torch
import torch.nn as nn

from .config import (
    NUM_CLASSES,
    HIDDEN_WIDTH,
    DROPOUT
)


# --------------------------------------------------
# 1. RESIDUAL BLOCK
# --------------------------------------------------

class ResidualBlock(nn.Module):
    """
    A residual block with two fully connected layers.
    """

    def __init__(self, width, dropout=DROPOUT):

        super().__init__()

        self.block = nn.Sequential(

            nn.Linear(width, width),

            nn.BatchNorm1d(width),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(width, width),

            nn.BatchNorm1d(width)
        )

        self.activation = nn.GELU()

    def forward(self, x):

        # Add original input to the block output.
        output = self.block(x)

        return self.activation(x + output)


# --------------------------------------------------
# 2. DEEP RESIDUAL IDS NETWORK
# --------------------------------------------------

class DeepResIDSNet(nn.Module):
    """
    Deep residual neural network for
    seven-class intrusion detection.
    """

    def __init__(
        self,
        input_dim,
        num_classes=NUM_CLASSES,
        width=HIDDEN_WIDTH
    ):

        super().__init__()

        # Input projection layer
        self.input_proj = nn.Sequential(

            nn.Linear(input_dim, width),

            nn.BatchNorm1d(width),

            nn.GELU()
        )

        # Residual blocks
        self.res1 = ResidualBlock(width)

        self.res2 = ResidualBlock(width)

        # Final classification layers
        self.head = nn.Sequential(

            nn.Linear(width, 256),

            nn.BatchNorm1d(256),

            nn.GELU(),

            nn.Dropout(0.15),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        x = self.input_proj(x)

        x = self.res1(x)

        x = self.res2(x)

        return self.head(x)


# --------------------------------------------------
# 3. MODEL TESTING
# --------------------------------------------------

if __name__ == "__main__":

    print("Deep Residual IDS Model Test")

    # CICIDS2017 input feature count is determined
    # later from the actual dataset.
    input_features = 78

    model = DeepResIDSNet(
        input_dim=input_features
    )

    # Create a sample batch of four records.
    sample = torch.randn(4, input_features)

    # Run the model
    model.eval()

    with torch.no_grad():

        output = model(sample)

    # Count model parameters
    parameters = sum(
        p.numel() for p in model.parameters()
    )

    print("Input Shape:", sample.shape)

    print("Output Shape:", output.shape)

    print("Number of Classes:", NUM_CLASSES)

    print("Model Parameters:", f"{parameters:,}")

    print("\nModel is working successfully!")
