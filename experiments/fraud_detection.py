#!/usr/bin/env python3
"""
experiments/fraud_detection.py - Tabular fraud detection with an MLP.
Simulates an imbalanced transaction dataset and trains a neural network classifier
using BCEWithLogitsLoss and Adam.
"""

import sys
import os
import numpy as np

# Add repository root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import Tensor, Sequential, Linear, ReLU, BCEWithLogitsLoss, Adam, no_grad


def generate_fraud_dataset(
    n_samples: int = 2000,
    n_features: int = 10,
    fraud_ratio: float = 0.05,
    seed: int = 42,
):
    """
    Generate synthetic credit card transaction dataset with imbalanced fraud labels.
    """
    np.random.seed(seed)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # Legitimate transactions: Gaussian cluster around 0
    X_legit = np.random.normal(loc=0.0, scale=1.0, size=(n_legit, n_features))
    y_legit = np.zeros(n_legit, dtype=np.float32)

    # Fraudulent transactions: Shifted mean and wider variance
    fraud_shift = np.random.uniform(1.5, 3.0, size=(n_features,))
    X_fraud = np.random.normal(loc=fraud_shift, scale=1.5, size=(n_fraud, n_features))
    y_fraud = np.ones(n_fraud, dtype=np.float32)

    X = np.vstack([X_legit, X_fraud]).astype(np.float32)
    y = np.concatenate([y_legit, y_fraud])

    # Shuffle
    indices = np.random.permutation(n_samples)
    X, y = X[indices], y[indices]

    # Standardize features
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-8
    X = (X - mean) / std

    # Train / test split (80% / 20%)
    split = int(0.8 * n_samples)
    return X[:split], y[:split], X[split:], y[split:]


def calculate_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    """Compute precision, recall, f1, and accuracy."""
    y_pred = (y_prob >= threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def main():
    print("=" * 65)
    print("backprop_from_bedrock: Fraud Detection Experiment")
    print("Task: Imbalanced Tabular Binary Classification (5% Fraud)")
    print("=" * 65)

    X_train, y_train, X_test, y_test = generate_fraud_dataset(
        n_samples=2000, n_features=10, fraud_ratio=0.05, seed=42
    )
    print(f"Dataset: {len(X_train)} train samples, {len(X_test)} test samples (10 features)")
    print(f"Train fraud cases: {int(np.sum(y_train))} ({np.mean(y_train)*100:.1f}%)")
    print(f"Test fraud cases:  {int(np.sum(y_test))} ({np.mean(y_test)*100:.1f}%)\n")

    # Define MLP model: 10 -> 32 -> 16 -> 1
    np.random.seed(42)
    model = Sequential(
        Linear(10, 32),
        ReLU(),
        Linear(32, 16),
        ReLU(),
        Linear(16, 1),
    )

    criterion = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=0.01)

    X_train_t = Tensor(X_train)
    y_train_t = Tensor(y_train[:, None])

    epochs = 80
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == 1:
            with no_grad():
                probs = 1.0 / (1.0 + np.exp(-np.clip(logits.data.squeeze(), -500, 500)))
                metrics = calculate_metrics(y_train, probs)
                print(
                    f"Epoch {epoch:2d}/{epochs} | Loss: {loss.item():.4f} | "
                    f"Prec: {metrics['precision']:.3f} | Rec: {metrics['recall']:.3f} | F1: {metrics['f1']:.3f}"
                )

    # Evaluate on test set
    with no_grad():
        test_logits = model(Tensor(X_test))
        test_probs = 1.0 / (1.0 + np.exp(-np.clip(test_logits.data.squeeze(), -500, 500)))
        test_metrics = calculate_metrics(y_test, test_probs)

    print("\n" + "=" * 65)
    print("Test Set Performance:")
    print(f"  Accuracy:  {test_metrics['accuracy'] * 100:.2f}%")
    print(f"  Precision: {test_metrics['precision']:.3f}")
    print(f"  Recall:    {test_metrics['recall']:.3f}")
    print(f"  F1-Score:  {test_metrics['f1']:.3f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
