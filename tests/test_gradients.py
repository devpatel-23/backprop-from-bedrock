"""
tests/test_gradients.py - Numerical gradient checks and PyTorch reference tests.
"""

from typing import Callable, Sequence, List
import numpy as np
import pytest
import torch

from core.tensor import Tensor
from core.autograd import no_grad
from core.layers import Linear, ReLU, Sigmoid, Tanh, Softmax, Sequential, Parameter
from core.losses import MSELoss, BinaryCrossEntropyLoss, BCEWithLogitsLoss, CrossEntropyLoss
from core.optimizers import SGD, Adam


def numerical_gradient(eval_fn: Callable[[], float], x: Tensor, eps: float = 1e-5) -> np.ndarray:
    """Compute numerical gradient using central differences."""
    orig_data = x.data.copy()
    grad_num = np.zeros_like(orig_data, dtype=np.float64)
    it = np.nditer(orig_data, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        old_val = float(orig_data[idx])

        x.data[idx] = old_val + eps
        y_plus = eval_fn()

        x.data[idx] = old_val - eps
        y_minus = eval_fn()

        x.data[idx] = old_val
        grad_num[idx] = (y_plus - y_minus) / (2.0 * eps)
        it.iternext()
    return grad_num


def check_gradient(
    fn: Callable[[], Tensor],
    tensors: Sequence[Tensor],
    eps: float = 1e-5,
    rtol: float = 1e-6,
) -> None:
    """
    Verify analytical backward gradients against finite-difference numerical gradients.
    Requires relative error < rtol for float64 tensors.
    """
    # 1. Forward and backward to get analytical gradients
    for t in tensors:
        t.zero_grad()
    loss = fn()
    loss.backward()

    # 2. Check each tensor against central difference numerical gradients
    for idx, t in enumerate(tensors):
        assert t.grad is not None, f"Tensor {idx} has None grad after backward."
        g_analytic = t.grad.copy()

        def eval_fn() -> float:
            with no_grad():
                val = fn()
                return float(val.data.squeeze())

        g_num = numerical_gradient(eval_fn, t, eps=eps)

        diff = np.linalg.norm(g_analytic - g_num)
        denom = max(np.linalg.norm(g_analytic), np.linalg.norm(g_num), 1e-12)
        rel_err = diff / denom

        assert rel_err < rtol, (
            f"Gradient check failed for tensor {idx}! rel_err={rel_err:.2e} >= {rtol:.2e}\n"
            f"Analytic:\n{g_analytic}\nNumerical:\n{g_num}"
        )


# =============================================================================
# 1. Elementary Tensor Operations Gradient Checks
# =============================================================================

def test_grad_add():
    np.random.seed(42)
    x = Tensor(np.random.randn(3, 4), requires_grad=True, dtype=np.float64)
    y = Tensor(np.random.randn(3, 4), requires_grad=True, dtype=np.float64)
    check_gradient(lambda: (x + y).sum(), [x, y])


def test_grad_add_broadcasting():
    np.random.seed(42)
    x = Tensor(np.random.randn(3, 4), requires_grad=True, dtype=np.float64)
    y = Tensor(np.random.randn(4), requires_grad=True, dtype=np.float64)
    check_gradient(lambda: (x + y).sum(), [x, y])

    z = Tensor(np.random.randn(1, 4), requires_grad=True, dtype=np.float64)
    check_gradient(lambda: (x + z).sum(), [x, z])


def test_grad_sub():
    np.random.seed(42)
    x = Tensor(np.random.randn(3, 4), requires_grad=True, dtype=np.float64)
    y = Tensor(np.random.randn(3, 4), requires_grad=True, dtype=np.float64)
    check_gradient(lambda: (x - y).sum(), [x, y])


def test_grad_sub_broadcasting():
    np.random.seed(42)
    x = Tensor(np.random.randn(3, 4), requires_grad=True, dtype=np.float64)
    y = Tensor(np.random.randn(4), requires_grad=True, dtype=np.float64)
    check_gradient(lambda: (x - y).sum(), [x, y])


def test_grad_neg():
    np.random.seed(42)
    x = Tensor(np.random.randn(3, 4), requires_grad=True, dtype=np.float64)
    check_gradient(lambda: (-x).sum(), [x])


def test_grad_mul():
    np.random.seed(42)
    x = Tensor(np.random.randn(3, 4), requires_grad=True, dtype=np.float64)
    y = Tensor(np.random.randn(3, 4), requires_grad=True, dtype=np.float64)
    check_gradient(lambda: (x * y).sum(), [x, y])


def test_grad_mul_broadcasting():
    np.random.seed(42)
    x = Tensor(np.random.randn(3, 4), requires_grad=True, dtype=np.float64)
    y = Tensor(np.random.randn(4), requires_grad=True, dtype=np.float64)
    check_gradient(lambda: (x * y).sum(), [x, y])


def test_grad_div():
    np.random.seed(42)
    x = Tensor(np.random.randn(3, 4), requires_grad=True, dtype=np.float64)
    # Ensure y is away from 0
    y = Tensor(np.random.uniform(1.0, 3.0, size=(3, 4)), requires_grad=True, dtype=np.float64)
    check_gradient(lambda: (x / y).sum(), [x, y])


def test_grad_pow_scalar():
    np.random.seed(42)
    x = Tensor(np.random.uniform(0.5, 3.0, size=(3, 4)), requires_grad=True, dtype=np.float64)
    check_gradient(lambda: (x ** 3).sum(), [x])


def test_grad_pow_tensor():
    np.random.seed(42)
    x = Tensor(np.random.uniform(0.5, 3.0, size=(3, 4)), requires_grad=True, dtype=np.float64)
    p = Tensor(np.random.uniform(1.0, 2.5, size=(3, 4)), requires_grad=True, dtype=np.float64)
    check_gradient(lambda: (x ** p).sum(), [x, p])


def test_grad_matmul_2d():
    np.random.seed(42)
    x = Tensor(np.random.randn(3, 5), requires_grad=True, dtype=np.float64)
    y = Tensor(np.random.randn(5, 4), requires_grad=True, dtype=np.float64)
    check_gradient(lambda: (x @ y).sum(), [x, y])


def test_grad_matmul_1d_2d():
    np.random.seed(42)
    x = Tensor(np.random.randn(5), requires_grad=True, dtype=np.float64)
    y = Tensor(np.random.randn(5, 4), requires_grad=True, dtype=np.float64)
    check_gradient(lambda: (x @ y).sum(), [x, y])


def test_grad_matmul_2d_1d():
    np.random.seed(42)
    x = Tensor(np.random.randn(3, 5), requires_grad=True, dtype=np.float64)
    y = Tensor(np.random.randn(5), requires_grad=True, dtype=np.float64)
    check_gradient(lambda: (x @ y).sum(), [x, y])


def test_grad_sum_all_axes():
    np.random.seed(42)
    x = Tensor(np.random.randn(2, 3, 4), requires_grad=True, dtype=np.float64)
    check_gradient(lambda: x.sum(), [x])


def test_grad_sum_axis():
    np.random.seed(42)
    x = Tensor(np.random.randn(2, 3, 4), requires_grad=True, dtype=np.float64)
    check_gradient(lambda: x.sum(axis=1).sum(), [x])
    check_gradient(lambda: x.sum(axis=(0, 2), keepdims=True).sum(), [x])


def test_grad_mean():
    np.random.seed(42)
    x = Tensor(np.random.randn(3, 4), requires_grad=True, dtype=np.float64)
    check_gradient(lambda: x.mean(), [x])
    check_gradient(lambda: x.mean(axis=1).sum(), [x])


def test_grad_reshape():
    np.random.seed(42)
    x = Tensor(np.random.randn(2, 6), requires_grad=True, dtype=np.float64)
    check_gradient(lambda: x.reshape(3, 4).sum(), [x])


def test_grad_transpose():
    np.random.seed(42)
    x = Tensor(np.random.randn(2, 3, 4), requires_grad=True, dtype=np.float64)
    check_gradient(lambda: x.transpose(2, 0, 1).sum(), [x])
    check_gradient(lambda: x.reshape(6, 4).T.sum(), [x])


# =============================================================================
# 2. Activation Functions Gradient Checks
# =============================================================================

def test_grad_relu():
    np.random.seed(42)
    # Non-zero entries to avoid non-differentiable point at 0
    data = np.array([[-2.0, -0.5, 0.5, 2.0], [1.5, -1.0, 3.0, -2.5]], dtype=np.float64)
    x = Tensor(data, requires_grad=True, dtype=np.float64)
    check_gradient(lambda: x.relu().sum(), [x])


def test_grad_sigmoid():
    np.random.seed(42)
    x = Tensor(np.random.randn(3, 4), requires_grad=True, dtype=np.float64)
    check_gradient(lambda: x.sigmoid().sum(), [x])


def test_grad_tanh():
    np.random.seed(42)
    x = Tensor(np.random.randn(3, 4), requires_grad=True, dtype=np.float64)
    check_gradient(lambda: x.tanh().sum(), [x])


def test_grad_softmax():
    np.random.seed(42)
    x = Tensor(np.random.randn(3, 4), requires_grad=True, dtype=np.float64)
    v = np.random.randn(3, 4)
    check_gradient(lambda: (x.softmax(axis=-1) * v).sum(), [x])


# =============================================================================
# 3. Layers & Modules Gradient Checks
# =============================================================================

def test_grad_linear():
    np.random.seed(42)
    layer = Linear(in_features=4, out_features=3, dtype=np.float64)
    x = Tensor(np.random.randn(2, 4), requires_grad=True, dtype=np.float64)
    params = [x, layer.weight, layer.bias]
    check_gradient(lambda: layer(x).sum(), params)


def test_grad_sequential_mlp():
    np.random.seed(42)
    l1 = Linear(3, 5, dtype=np.float64)
    act = Tanh()
    l2 = Linear(5, 2, dtype=np.float64)
    model = Sequential(l1, act, l2)

    x = Tensor(np.random.randn(4, 3), requires_grad=True, dtype=np.float64)
    params = [x] + model.parameters()
    check_gradient(lambda: model(x).sum(), params)


# =============================================================================
# 4. Loss Functions Gradient Checks
# =============================================================================

def test_grad_mse_loss():
    np.random.seed(42)
    pred = Tensor(np.random.randn(4, 3), requires_grad=True, dtype=np.float64)
    target = np.random.randn(4, 3)
    loss_fn = MSELoss()
    check_gradient(lambda: loss_fn(pred, target), [pred])


def test_grad_bce_loss():
    np.random.seed(42)
    # Probabilities in (0.1, 0.9)
    pred = Tensor(np.random.uniform(0.1, 0.9, size=(4, 2)), requires_grad=True, dtype=np.float64)
    target = np.random.randint(0, 2, size=(4, 2)).astype(np.float64)
    loss_fn = BinaryCrossEntropyLoss()
    check_gradient(lambda: loss_fn(pred, target), [pred])


def test_grad_bce_with_logits_loss():
    np.random.seed(42)
    logits = Tensor(np.random.randn(4, 2), requires_grad=True, dtype=np.float64)
    target = np.random.randint(0, 2, size=(4, 2)).astype(np.float64)
    loss_fn = BCEWithLogitsLoss()
    check_gradient(lambda: loss_fn(logits, target), [logits])


def test_grad_cross_entropy_loss_indices():
    np.random.seed(42)
    logits = Tensor(np.random.randn(4, 5), requires_grad=True, dtype=np.float64)
    target = np.random.randint(0, 5, size=(4,))
    loss_fn = CrossEntropyLoss()
    check_gradient(lambda: loss_fn(logits, target), [logits])


def test_grad_cross_entropy_loss_soft():
    np.random.seed(42)
    logits = Tensor(np.random.randn(4, 5), requires_grad=True, dtype=np.float64)
    # Soft probability targets
    t_raw = np.random.uniform(0.1, 1.0, size=(4, 5))
    target = t_raw / np.sum(t_raw, axis=-1, keepdims=True)
    loss_fn = CrossEntropyLoss()
    check_gradient(lambda: loss_fn(logits, target), [logits])


# =============================================================================
# 5. PyTorch Equivalence Tests
# =============================================================================

def test_vs_pytorch_linear():
    np.random.seed(42)
    w_init = np.random.randn(4, 3).astype(np.float64)
    b_init = np.random.randn(3).astype(np.float64)
    x_init = np.random.randn(2, 4).astype(np.float64)

    # Ours
    linear = Linear(4, 3, dtype=np.float64)
    linear.weight.data = w_init.copy()
    linear.bias.data = b_init.copy()
    x = Tensor(x_init.copy(), requires_grad=True, dtype=np.float64)
    out = linear(x).sum()
    out.backward()

    # PyTorch
    x_pt = torch.tensor(x_init, dtype=torch.float64, requires_grad=True)
    w_pt = torch.tensor(w_init, dtype=torch.float64, requires_grad=True)
    b_pt = torch.tensor(b_init, dtype=torch.float64, requires_grad=True)
    out_pt = (x_pt @ w_pt + b_pt).sum()
    out_pt.backward()

    np.testing.assert_allclose(x.grad, x_pt.grad.numpy(), rtol=1e-10)
    np.testing.assert_allclose(linear.weight.grad, w_pt.grad.numpy(), rtol=1e-10)
    np.testing.assert_allclose(linear.bias.grad, b_pt.grad.numpy(), rtol=1e-10)


def test_vs_pytorch_cross_entropy():
    np.random.seed(42)
    logits_np = np.random.randn(5, 4).astype(np.float64)
    target_np = np.random.randint(0, 4, size=(5,))

    # Ours
    logits = Tensor(logits_np.copy(), requires_grad=True, dtype=np.float64)
    loss_fn = CrossEntropyLoss()
    loss = loss_fn(logits, target_np)
    loss.backward()

    # PyTorch
    logits_pt = torch.tensor(logits_np, dtype=torch.float64, requires_grad=True)
    target_pt = torch.tensor(target_np, dtype=torch.long)
    loss_pt = torch.nn.functional.cross_entropy(logits_pt, target_pt)
    loss_pt.backward()

    np.testing.assert_allclose(loss.item(), loss_pt.item(), rtol=1e-10)
    np.testing.assert_allclose(logits.grad, logits_pt.grad.numpy(), rtol=1e-10)


# =============================================================================
# 6. Optimizer Update Tests
# =============================================================================

def test_sgd_optimizer():
    # Minimize f(w) = 0.5 * (w - 3)^2
    w = Parameter(np.array([10.0]))
    opt = SGD([w], lr=0.1)

    initial_loss = 0.5 * (w - 3.0) ** 2
    for _ in range(100):
        opt.zero_grad()
        loss = 0.5 * (w - 3.0) ** 2
        loss.backward()
        opt.step()

    final_loss = 0.5 * (w - 3.0) ** 2
    assert final_loss.item() < initial_loss.item()
    np.testing.assert_allclose(w.data, [3.0], atol=1e-3)


def test_adam_optimizer():
    # Minimize f(w) = 0.5 * (w - 5)^2
    w = Parameter(np.array([0.0]))
    opt = Adam([w], lr=0.1)

    initial_loss = 0.5 * (w - 5.0) ** 2
    for _ in range(150):
        opt.zero_grad()
        loss = 0.5 * (w - 5.0) ** 2
        loss.backward()
        opt.step()

    final_loss = 0.5 * (w - 5.0) ** 2
    assert final_loss.item() < initial_loss.item()
    np.testing.assert_allclose(w.data, [5.0], atol=1e-2)
