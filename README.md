# backprop_from_bedrock

A modular, from-scratch deep learning library and reverse-mode automatic differentiation (autograd) engine built on pure NumPy and standard Python.

`backprop_from_bedrock` demonstrates the foundational mechanics of modern neural network frameworks (like PyTorch and Micrograd) scaled to multi-dimensional tensors, full broadcasting, matrix calculus, standard neural network modules, activations, numerically stabilized loss functions, and first-order optimizers.

---

## Table of Contents

- [Features](#features)
- [Architecture & Repository Structure](#architecture--repository-structure)
- [Mathematical Foundations](#mathematical-foundations)
  - [1. Tensor Operations & Broadcasting](#1-tensor-operations--broadcasting)
  - [2. Linear (Dense) Layer](#2-linear-dense-layer)
  - [3. Activation Functions](#3-activation-functions)
  - [4. Loss Functions & Numerical Stability](#4-loss-functions--numerical-stability)
  - [5. Optimizers](#5-optimizers)
- [Installation & Quickstart](#installation--quickstart)
- [Running Tests & Numerical Gradient Checking](#running-tests--numerical-gradient-checking)
- [End-to-End Demo](#end-to-end-demo)
- [Experiments & Benchmarks](#experiments--benchmarks)
  - [Tabular Fraud Detection Experiment](#tabular-fraud-detection-experiment)
  - [Benchmark vs PyTorch (CPU)](#benchmark-vs-pytorch-cpu)

---

## Features

- **Dynamic Reverse-Mode Autograd**: DAG construction with topological sort and local backward gradient closures.
- **Multi-Dimensional Broadcasting**: `unbroadcast` engine maps upstream gradients back to operand shapes across arbitrary broadcasting dimensions.
- **Pure NumPy Core**: No black-box autograd frameworks or heavy dependencies used for core mathematical execution.
- **Rigorous Gradient Verification**: Every operator, layer, activation, and loss verified via finite-difference numerical gradient checks ($\epsilon = 10^{-5}$, relative error $< 10^{-6}$ on `float64`) and cross-checked against PyTorch autograd.
- **Numerically Stabilized Implementations**: Log-Sum-Exp cross-entropy and clipped/log-domain binary cross-entropy prevent overflow and underflow.
- **PyTorch-Style Modular API**: Includes `Tensor`, `Parameter`, `Module`, `Linear`, `Sequential`, activations (`ReLU`, `Sigmoid`, `Tanh`, `Softmax`), losses (`MSELoss`, `CrossEntropyLoss`, `BCEWithLogitsLoss`), and optimizers (`SGD`, `Adam`).

---

## Architecture & Repository Structure

```
backprop-from-bedrock/
├── core/
│   ├── __init__.py         # Public API exports
│   ├── autograd.py         # Graph traversal, topological sort, unbroadcast, no_grad
│   ├── tensor.py           # Multi-dimensional Tensor node with chain-rule closures
│   ├── layers.py           # Parameter, Module, Linear, activations, Sequential
│   ├── losses.py           # MSELoss, BCELoss, BCEWithLogitsLoss, CrossEntropyLoss
│   └── optimizers.py       # SGD (with momentum & Nesterov), Adam
├── examples/
│   └── train_demo.py       # End-to-end 3-class spiral classification training demo
├── experiments/
│   └── fraud_detection.py  # Imbalanced tabular financial fraud classification experiment
├── benchmarks/
│   └── vs_pytorch.py       # Speed and numerical accuracy benchmark vs PyTorch CPU
├── notebooks/
│   └── derivations.ipynb   # Interactive Jupyter notebook with math derivations
├── tests/
│   └── test_gradients.py   # Unit test suite & numerical gradient check suite
├── pytest.ini              # Pytest configuration
├── requirements.txt        # Python dependencies
├── STATUS.md               # Project audit, roadmap, and implementation ledger
└── README.md
```

---

## Mathematical Foundations

### 1. Tensor Operations & Broadcasting

During forward evaluation, binary operations broadcast shapes $A$ and $B$ to an output shape $C$. During backpropagation, the upstream gradient $\frac{\partial L}{\partial C}$ must be reduced back to shapes $A$ and $B$:

$$\text{unbroadcast}(G, \text{shape}) = \sum_{\text{broadcasted axes}} G$$

- **Addition**: $Z = X + Y \implies \frac{\partial L}{\partial X} = \text{unbroadcast}\left(\frac{\partial L}{\partial Z}, X.\text{shape}\right)$
- **Multiplication**: $Z = X \odot Y \implies \frac{\partial L}{\partial X} = \text{unbroadcast}\left(\frac{\partial L}{\partial Z} \odot Y, X.\text{shape}\right)$
- **Division**: $Z = \frac{X}{Y} \implies \frac{\partial L}{\partial X} = \text{unbroadcast}\left(\frac{\frac{\partial L}{\partial Z}}{Y}, X.\text{shape}\right), \quad \frac{\partial L}{\partial Y} = \text{unbroadcast}\left(-\frac{\partial L}{\partial Z} \odot \frac{X}{Y^2}, Y.\text{shape}\right)$
- **Power**: $Z = X^p \implies \frac{\partial L}{\partial X} = \frac{\partial L}{\partial Z} \odot p X^{p-1}$

### 2. Linear (Dense) Layer

Given batch $X \in \mathbb{R}^{B \times D_{in}}$, weight matrix $W \in \mathbb{R}^{D_{in} \times D_{out}}$, and bias $b \in \mathbb{R}^{D_{out}}$:

$$Z = X W + b \quad \in \mathbb{R}^{B \times D_{out}}$$

Given upstream gradient $dZ = \frac{\partial L}{\partial Z} \in \mathbb{R}^{B \times D_{out}}$:
- $\frac{\partial L}{\partial X} = dZ \cdot W^T \in \mathbb{R}^{B \times D_{in}}$
- $\frac{\partial L}{\partial W} = X^T \cdot dZ \in \mathbb{R}^{D_{in} \times D_{out}}$
- $\frac{\partial L}{\partial b} = \sum_{i=1}^B dZ_{i, :} \in \mathbb{R}^{D_{out}}$

### 3. Activation Functions

- **ReLU**: $y = \max(0, x) \implies \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \odot \mathbf{1}_{\{x > 0\}}$
- **Sigmoid**: $\sigma(x) = \frac{1}{1 + e^{-x}} \implies \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \odot \sigma(x)(1 - \sigma(x))$
- **Tanh**: $y = \tanh(x) \implies \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \odot (1 - y^2)$
- **Softmax**: $p_i = \frac{e^{z_i}}{\sum_j e^{z_j}} \implies \frac{\partial L}{\partial z_i} = p_i \left( \frac{\partial L}{\partial p_i} - \sum_j \frac{\partial L}{\partial p_j} p_j \right)$

### 4. Loss Functions & Numerical Stability

#### Cross-Entropy with Logits (Log-Sum-Exp Trick)
For raw logits $z \in \mathbb{R}^{B \times C}$ and ground-truth class indices $y_i \in \{0, \dots, C-1\}$:

$$m_i = \max_c z_{ic}$$

$$\log p_{ic} = (z_{ic} - m_i) - \log \sum_{j=1}^C e^{z_{ij} - m_i}$$

$$L = -\frac{1}{B} \sum_{i=1}^B \log p_{i, y_i}$$

The analytical gradient with respect to logits $z_{ic}$ is remarkably simple and immune to float underflow/overflow:

$$\frac{\partial L}{\partial z_{ic}} = \frac{1}{B} \left( p_{ic} - \mathbf{1}_{\{c = y_i\}} \right)$$

#### Binary Cross-Entropy with Logits
Combines sigmoid and binary cross-entropy:

$$L_i = \max(z_i, 0) - z_i y_i + \log\left(1 + e^{-|z_i|}\right)$$

$$\frac{\partial L_i}{\partial z_i} = \sigma(z_i) - y_i$$

### 5. Optimizers

- **SGD**:
  $$v_t = \mu v_{t-1} + g_t + \lambda \theta_t$$
  $$\theta_{t+1} = \theta_t - \eta v_t$$
- **Adam**:
  $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
  $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
  $$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

---

## Installation & Quickstart

Clone the repository and install requirements:

```bash
git clone https://github.com/devpatel-23/backprop-from-bedrock.git
cd backprop-from-bedrock
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Running Tests & Numerical Gradient Checking

The test suite runs finite-difference central-difference gradient checks:

$$g_{\text{num}} = \frac{f(x + \epsilon) - f(x - \epsilon)}{2 \epsilon}, \quad \epsilon = 10^{-5}$$

Asserting relative error $\frac{\|g_{\text{analytic}} - g_{\text{num}}\|_2}{\max(\|g_{\text{analytic}}\|_2, \|g_{\text{num}}\|_2, 10^{-12})} < 10^{-6}$ on `float64`.

Run the full pytest suite:

```bash
pytest tests/test_gradients.py -v
```

Output:
```
============================== 33 passed in 1.44s ==============================
```

---

## End-to-End Demo

Run the standalone 3-class non-linear spiral training demonstration:

```bash
python examples/train_demo.py
```

Expected output:
```
============================================================
backprop_from_bedrock: End-to-End MLP Training Demo
Dataset: 3-Class Non-Linear Spiral (300 samples, 2 features)
============================================================
Model parameters: 675
Training for 100 epochs with Adam optimizer (lr=0.02)...

Epoch   1/100 | Loss: 1.0909 | Accuracy:  33.3%
Epoch  10/100 | Loss: 0.7689 | Accuracy:  53.7%
Epoch  20/100 | Loss: 0.5445 | Accuracy:  66.0%
Epoch  30/100 | Loss: 0.3278 | Accuracy:  85.7%
Epoch  40/100 | Loss: 0.1871 | Accuracy:  95.0%
Epoch  50/100 | Loss: 0.1152 | Accuracy:  97.7%
Epoch  60/100 | Loss: 0.0769 | Accuracy:  98.3%
Epoch  70/100 | Loss: 0.0579 | Accuracy:  98.3%
Epoch  80/100 | Loss: 0.0478 | Accuracy:  98.7%
Epoch  90/100 | Loss: 0.0414 | Accuracy:  99.3%
Epoch 100/100 | Loss: 0.0368 | Accuracy:  99.3%

============================================================
Training Complete! Final Loss: 0.0368 | Final Accuracy: 99.3%
============================================================
```

---

## Experiments & Benchmarks

### Tabular Fraud Detection Experiment

Trains an MLP on a synthetic imbalanced credit card transaction dataset (5% fraudulent transactions) with standardization and `BCEWithLogitsLoss`:

```bash
python experiments/fraud_detection.py
```

Sample output:
```
=================================================================
Test Set Performance:
  Accuracy:  99.75%
  Precision: 1.000
  Recall:    0.947
  F1-Score:  0.973
=================================================================
```

### Benchmark vs PyTorch (CPU)

Compares forward and backward pass speed and numerical gradient accuracy against PyTorch CPU:

```bash
python benchmarks/vs_pytorch.py
```

Sample benchmark output:
```
======================================================================
Benchmark: backprop_from_bedrock vs PyTorch (CPU)
Network: 64 -> 128 -> 128 -> 10 (MLP with ReLU)
Batch size: 128 | Iterations: 200
======================================================================
Numerical Verification:
  Max Output Difference:   0.00e+00
  Max Weight1 Grad Diff:   0.00e+00

Runtime Benchmark Results:
  backprop_from_bedrock: 0.387 ms/step  (2580.9 steps/sec)
  PyTorch CPU (C++):     0.216 ms/step  (4627.5 steps/sec)
  Relative Ratio:        1.79x of PyTorch CPU runtime
======================================================================
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
