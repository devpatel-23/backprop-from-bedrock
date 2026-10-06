"""
core/autograd.py - Automatic differentiation engine and graph traversal.
"""

from typing import Tuple, List, Set, Optional, Callable, Any
import functools
import numpy as np

# Global autograd state
_GRAD_ENABLED: bool = True


def is_grad_enabled() -> bool:
    """Return True if autograd is currently tracking gradients."""
    return _GRAD_ENABLED


def set_grad_enabled(mode: bool) -> bool:
    """
    Enable or disable gradient calculation.
    Returns the previous state.
    """
    global _GRAD_ENABLED
    prev = _GRAD_ENABLED
    _GRAD_ENABLED = mode
    return prev


class no_grad:
    """
    Context manager and decorator that disables gradient calculation.
    Saves computation during evaluation/inference.
    """

    def __enter__(self) -> "no_grad":
        self.prev = set_grad_enabled(False)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        set_grad_enabled(self.prev)

    def __call__(self, fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self:
                return fn(*args, **kwargs)
        return wrapper


def unbroadcast(grad: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Sum gradients along dimensions that were broadcasted during forward pass
    to match the original target_shape.
    """
    if grad.shape == target_shape:
        return grad

    grad_ndim = grad.ndim
    target_ndim = len(target_shape)
    leading_dims = grad_ndim - target_ndim

    # Sum out any extra prepended dimensions created by broadcasting
    for _ in range(leading_dims):
        grad = grad.sum(axis=0)

    # Sum along axes where target_shape has length 1 but grad has length > 1
    for axis, (g_dim, t_dim) in enumerate(zip(grad.shape, target_shape)):
        if t_dim == 1 and g_dim > 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad.reshape(target_shape)


def topological_sort(root: Any) -> List[Any]:
    """
    Construct a reverse topological ordering of the computation graph DAG.
    """
    topo: List[Any] = []
    visited: Set[int] = set()

    def build_topo(node: Any) -> None:
        node_id = id(node)
        if node_id not in visited:
            visited.add(node_id)
            for child in getattr(node, "_prev", ()):
                build_topo(child)
            topo.append(node)

    build_topo(root)
    return topo


def backward(root: Any, grad: Optional[np.ndarray] = None) -> None:
    """
    Compute gradients of all graph ancestors with respect to the root tensor.
    """
    if not root.requires_grad:
        return

    # Seed the gradient at root: 1.0 for scalars or all-ones if shape > 0
    if grad is None:
        if root.shape == () or root.shape == (1,):
            root.grad = np.ones_like(root.data)
        else:
            raise RuntimeError(
                f"grad must be specified for non-scalar root tensor with shape {root.shape}"
            )
    else:
        root.grad = np.array(grad, dtype=root.dtype, copy=True)

    # Order nodes so every consumer runs before its dependencies
    ordered = reversed(topological_sort(root))
    for node in ordered:
        node._backward()
