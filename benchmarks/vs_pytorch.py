#!/usr/bin/env python3
"""
benchmarks/vs_pytorch.py - Runtime and accuracy benchmarking vs PyTorch CPU.
Measures forward pass time, backward pass time, and end-to-end training throughput.
"""

import sys
import os
import time
import numpy as np
import torch

# Add repository root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import Tensor, Sequential, Linear, ReLU, MSELoss


def run_benchmark(batch_size: int = 128, in_dim: int = 64, hidden_dim: int = 128, out_dim: int = 10, iters: int = 200):
    print("=" * 70)
    print("Benchmark: backprop_from_bedrock vs PyTorch (CPU)")
    print(f"Network: {in_dim} -> {hidden_dim} -> {hidden_dim} -> {out_dim} (MLP with ReLU)")
    print(f"Batch size: {batch_size} | Iterations: {iters}")
    print("=" * 70)

    # Fixed seed for fair initializations
    np.random.seed(42)
    w1_np = np.random.randn(in_dim, hidden_dim).astype(np.float32) * 0.05
    b1_np = np.zeros(hidden_dim, dtype=np.float32)
    w2_np = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.05
    b2_np = np.zeros(hidden_dim, dtype=np.float32)
    w3_np = np.random.randn(hidden_dim, out_dim).astype(np.float32) * 0.05
    b3_np = np.zeros(out_dim, dtype=np.float32)

    X_np = np.random.randn(batch_size, in_dim).astype(np.float32)
    Y_np = np.random.randn(batch_size, out_dim).astype(np.float32)

    # -------------------------------------------------------------------------
    # 1. backprop_from_bedrock setup
    # -------------------------------------------------------------------------
    model_ours = Sequential(
        Linear(in_dim, hidden_dim),
        ReLU(),
        Linear(hidden_dim, hidden_dim),
        ReLU(),
        Linear(hidden_dim, out_dim),
    )
    model_ours.layers[0].weight.data = w1_np.copy()
    model_ours.layers[0].bias.data = b1_np.copy()
    model_ours.layers[2].weight.data = w2_np.copy()
    model_ours.layers[2].bias.data = b2_np.copy()
    model_ours.layers[4].weight.data = w3_np.copy()
    model_ours.layers[4].bias.data = b3_np.copy()

    X_ours = Tensor(X_np.copy(), requires_grad=True)
    Y_ours = Tensor(Y_np.copy())
    criterion_ours = MSELoss()

    # -------------------------------------------------------------------------
    # 2. PyTorch CPU setup
    # -------------------------------------------------------------------------
    class PyTorchMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(in_dim, hidden_dim)
            self.l2 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.l3 = torch.nn.Linear(hidden_dim, out_dim)

        def forward(self, x):
            return self.l3(torch.relu(self.l2(torch.relu(self.l1(x)))))

    model_pt = PyTorchMLP()
    with torch.no_grad():
        model_pt.l1.weight.copy_(torch.from_numpy(w1_np.T))
        model_pt.l1.bias.copy_(torch.from_numpy(b1_np))
        model_pt.l2.weight.copy_(torch.from_numpy(w2_np.T))
        model_pt.l2.bias.copy_(torch.from_numpy(b2_np))
        model_pt.l3.weight.copy_(torch.from_numpy(w3_np.T))
        model_pt.l3.bias.copy_(torch.from_numpy(b3_np))

    X_pt = torch.tensor(X_np.copy(), requires_grad=True)
    Y_pt = torch.tensor(Y_np.copy())
    criterion_pt = torch.nn.MSELoss()

    # -------------------------------------------------------------------------
    # 3. Numerical Gradient & Output Equivalence Check
    # -------------------------------------------------------------------------
    out_ours = model_ours(X_ours)
    loss_ours = criterion_ours(out_ours, Y_ours)
    loss_ours.backward()

    out_pt = model_pt(X_pt)
    loss_pt = criterion_pt(out_pt, Y_pt)
    loss_pt.backward()

    output_diff = np.max(np.abs(out_ours.data - out_pt.detach().numpy()))
    grad_w1_diff = np.max(np.abs(model_ours.layers[0].weight.grad - model_pt.l1.weight.grad.numpy().T))
    print(f"Numerical Verification:")
    print(f"  Max Output Difference:   {output_diff:.2e}")
    print(f"  Max Weight1 Grad Diff:   {grad_w1_diff:.2e}\n")

    # -------------------------------------------------------------------------
    # 4. Benchmark backprop_from_bedrock
    # -------------------------------------------------------------------------
    # Warmup
    for _ in range(10):
        model_ours.zero_grad()
        loss = criterion_ours(model_ours(X_ours), Y_ours)
        loss.backward()

    t0 = time.perf_counter()
    for _ in range(iters):
        model_ours.zero_grad()
        loss = criterion_ours(model_ours(X_ours), Y_ours)
        loss.backward()
    ours_time = time.perf_counter() - t0
    ours_steps_per_sec = iters / ours_time

    # -------------------------------------------------------------------------
    # 5. Benchmark PyTorch CPU
    # -------------------------------------------------------------------------
    # Warmup
    for _ in range(10):
        model_pt.zero_grad()
        loss = criterion_pt(model_pt(X_pt), Y_pt)
        loss.backward()

    t0 = time.perf_counter()
    for _ in range(iters):
        model_pt.zero_grad()
        loss = criterion_pt(model_pt(X_pt), Y_pt)
        loss.backward()
    pt_time = time.perf_counter() - t0
    pt_steps_per_sec = iters / pt_time

    # -------------------------------------------------------------------------
    # Report Results
    # -------------------------------------------------------------------------
    print("Runtime Benchmark Results:")
    print(f"  backprop_from_bedrock: {ours_time * 1000 / iters:.3f} ms/step  ({ours_steps_per_sec:.1f} steps/sec)")
    print(f"  PyTorch CPU (C++):     {pt_time * 1000 / iters:.3f} ms/step  ({pt_steps_per_sec:.1f} steps/sec)")
    print(f"  Relative Ratio:        {ours_time / pt_time:.2f}x of PyTorch CPU runtime")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
