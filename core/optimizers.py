"""
core/optimizers.py - First-order parameter optimizers: SGD (with momentum) and Adam.
"""

from typing import Iterable, Tuple, Dict, Optional, List, Union
import numpy as np

from core.tensor import Tensor
from core.layers import Parameter


class Optimizer:
    """
    Base class for all optimizers.
    Manages parameters and provides gradient zeroing.
    """

    def __init__(
        self,
        params: Iterable[Union[Parameter, Tensor]],
        lr: float,
        weight_decay: float = 0.0,
    ) -> None:
        self.params: List[Union[Parameter, Tensor]] = list(params)
        self.lr: float = lr
        self.weight_decay: float = weight_decay

    def zero_grad(self) -> None:
        """Reset gradients for all managed parameters."""
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        """Perform a single parameter update step."""
        raise NotImplementedError("Subclasses of Optimizer must implement step().")


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer with optional momentum and Nesterov acceleration.
    """

    def __init__(
        self,
        params: Iterable[Union[Parameter, Tensor]],
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        super().__init__(params, lr=lr, weight_decay=weight_decay)
        self.momentum: float = momentum
        self.nesterov: bool = nesterov
        self.velocities: Dict[int, np.ndarray] = {}

    def step(self) -> None:
        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p.data

            p_id = id(p)
            if self.momentum != 0.0:
                if p_id not in self.velocities:
                    self.velocities[p_id] = np.zeros_like(p.data)

                v = self.momentum * self.velocities[p_id] + grad
                self.velocities[p_id] = v

                if self.nesterov:
                    step_dir = grad + self.momentum * v
                else:
                    step_dir = v
            else:
                step_dir = grad

            p.data = p.data - self.lr * step_dir.astype(p.dtype, copy=False)


class Adam(Optimizer):
    """
    Adam optimizer with first and second moment tracking and bias correction.
    """

    def __init__(
        self,
        params: Iterable[Union[Parameter, Tensor]],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr=lr, weight_decay=weight_decay)
        self.beta1, self.beta2 = betas
        self.eps: float = eps
        self.m: Dict[int, np.ndarray] = {}
        self.v: Dict[int, np.ndarray] = {}
        self.t: Dict[int, int] = {}

    def step(self) -> None:
        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p.data

            p_id = id(p)
            t = self.t.get(p_id, 0) + 1
            self.t[p_id] = t

            if p_id not in self.m:
                self.m[p_id] = np.zeros_like(p.data)
                self.v[p_id] = np.zeros_like(p.data)

            m = self.beta1 * self.m[p_id] + (1.0 - self.beta1) * grad
            v = self.beta2 * self.v[p_id] + (1.0 - self.beta2) * (grad ** 2)

            self.m[p_id] = m
            self.v[p_id] = v

            # Bias correction
            m_hat = m / (1.0 - self.beta1 ** t)
            v_hat = v / (1.0 - self.beta2 ** t)

            step_dir = m_hat / (np.sqrt(v_hat) + self.eps)
            p.data = p.data - self.lr * step_dir.astype(p.dtype, copy=False)
