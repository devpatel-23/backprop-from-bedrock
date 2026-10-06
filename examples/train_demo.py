#!/usr/bin/env python3
"""
examples/train_demo.py - End-to-end training demonstration.
Trains a multi-layer perceptron (MLP) on a non-linear 3-class spiral dataset
using pure NumPy autograd from backprop_from_bedrock.
"""

import sys
import os
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import Tensor, Sequential, Linear, ReLU, CrossEntropyLoss, Adam


def generate_spiral_data(samples_per_class: int = 100, num_classes: int = 3, seed: int = 42):
    """Generate synthetic non-linear 2D spiral classification dataset."""
    np.random.seed(seed)
    total_samples = samples_per_class * num_classes
    X = np.zeros((total_samples, 2), dtype=np.float32)
    y = np.zeros(total_samples, dtype=np.int32)

    for j in range(num_classes):
        ix = range(samples_per_class * j, samples_per_class * (j + 1))
        r = np.linspace(0.0, 1.0, samples_per_class)
        t = np.linspace(j * 4.0, (j + 1) * 4.0, samples_per_class) + np.random.randn(samples_per_class) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j

    return X, y


def main():
    print("=" * 60)
    print("backprop_from_bedrock: End-to-End MLP Training Demo")
    print("Dataset: 3-Class Non-Linear Spiral (300 samples, 2 features)")
    print("=" * 60)

    # 1. Prepare reproducible data
    X_train, y_train = generate_spiral_data(samples_per_class=100, num_classes=3, seed=42)
    X_tensor = Tensor(X_train)

    # 2. Build model architecture: 2 -> 32 -> 16 -> 3
    np.random.seed(42)
    model = Sequential(
        Linear(2, 32),
        ReLU(),
        Linear(32, 16),
        ReLU(),
        Linear(16, 3),
    )

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.02)

    epochs = 100
    print(f"Model parameters: {sum(p.data.size for p in model.parameters())}")
    print(f"Training for {epochs} epochs with Adam optimizer (lr=0.02)...\n")

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        logits = model(X_tensor)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            preds = np.argmax(logits.data, axis=1)
            accuracy = np.mean(preds == y_train) * 100.0
            print(f"Epoch {epoch:3d}/{epochs} | Loss: {loss.item():.4f} | Accuracy: {accuracy:5.1f}%")

    # Final evaluation
    final_preds = np.argmax(model(X_tensor).data, axis=1)
    final_acc = np.mean(final_preds == y_train) * 100.0
    print("\n" + "=" * 60)
    print(f"Training Complete! Final Loss: {loss.item():.4f} | Final Accuracy: {final_acc:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
