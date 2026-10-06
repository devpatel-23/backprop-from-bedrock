"""
core/losses.py - Standard loss functions with numerically stable forward/backward passes.
"""

from typing import Union, Optional, Any
import numpy as np

from core.tensor import Tensor, _accumulate_grad
from core.layers import Module
from core.autograd import is_grad_enabled


class Loss(Module):
    """Base class for all loss functions."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Invalid reduction '{reduction}'. Must be 'mean', 'sum', or 'none'.")
        self.reduction = reduction


class MSELoss(Loss):
    """
    Mean Squared Error loss: (pred - target)^2.
    """

    def forward(self, pred: Tensor, target: Union[Tensor, np.ndarray, list]) -> Tensor:
        target_t = target if isinstance(target, Tensor) else Tensor(target, dtype=pred.dtype)
        diff = pred - target_t
        squared = diff ** 2

        if self.reduction == "mean":
            return squared.mean()
        elif self.reduction == "sum":
            return squared.sum()
        return squared


class BinaryCrossEntropyLoss(Loss):
    """
    Binary Cross-Entropy loss between predicted probabilities and binary targets.
    Clips probabilities for numerical stability to prevent log(0).
    """

    def __init__(self, eps: float = 1e-15, reduction: str = "mean") -> None:
        super().__init__(reduction=reduction)
        self.eps = eps

    def forward(self, pred: Tensor, target: Union[Tensor, np.ndarray, list]) -> Tensor:
        target_t = target if isinstance(target, Tensor) else Tensor(target, dtype=pred.dtype)
        clipped_pred = pred.clip(self.eps, 1.0 - self.eps)

        term1 = target_t * clipped_pred.log()
        term2 = (1.0 - target_t) * (1.0 - clipped_pred).log()
        loss = -(term1 + term2)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


BCELoss = BinaryCrossEntropyLoss


class BCEWithLogitsLoss(Loss):
    """
    Numerically stable Binary Cross-Entropy loss with logits.
    Combines Sigmoid layer and BCELoss using the log-sum-exp trick:
    max(x, 0) - x * y + log(1 + exp(-|x|)).
    """

    def forward(self, logits: Tensor, target: Union[Tensor, np.ndarray, list]) -> Tensor:
        target_arr = target.data if isinstance(target, Tensor) else np.asarray(target, dtype=logits.dtype)
        z = logits.data

        # Numerically stable forward: max(z, 0) - z * y + log(1 + exp(-|z|))
        max_val = np.maximum(z, 0.0)
        loss_data = max_val - z * target_arr + np.log(1.0 + np.exp(-np.abs(z)))

        if self.reduction == "mean":
            loss_val = np.mean(loss_data)
        elif self.reduction == "sum":
            loss_val = np.sum(loss_data)
        else:
            loss_val = loss_data

        req_grad = is_grad_enabled() and logits.requires_grad
        out = Tensor(
            loss_val,
            requires_grad=req_grad,
            _children=(logits,) if req_grad else (),
            _op="BCEWithLogitsLoss",
            dtype=logits.dtype,
        )

        if req_grad:
            def _backward() -> None:
                # Gradient of BCEWithLogits w.r.t logits is sigmoid(z) - target
                sig_z = 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))
                d_logits = sig_z - target_arr

                if self.reduction == "mean":
                    scale = float(loss_data.size)
                    d_logits = d_logits * (out.grad / scale)
                elif self.reduction == "sum":
                    d_logits = d_logits * out.grad
                else:
                    d_logits = d_logits * out.grad

                _accumulate_grad(logits, d_logits)

            out._backward = _backward

        return out


class CrossEntropyLoss(Loss):
    """
    Multi-class Cross-Entropy loss with logits.
    Computes log-softmax followed by negative log likelihood in a single stable step.
    Supports both integer class targets (N,) and soft/one-hot probability targets (N, C).
    """

    def forward(self, logits: Tensor, target: Union[Tensor, np.ndarray, list]) -> Tensor:
        target_arr = target.data if isinstance(target, Tensor) else np.asarray(target)
        logits_data = logits.data

        # Numerically stable softmax probabilities via max subtraction
        m = np.max(logits_data, axis=-1, keepdims=True)
        exp_shifted = np.exp(logits_data - m)
        sum_exp = np.sum(exp_shifted, axis=-1, keepdims=True)
        probs = exp_shifted / sum_exp
        log_probs = (logits_data - m) - np.log(sum_exp)

        # Loss computation
        is_index_target = (target_arr.ndim == 1 and logits_data.ndim > 1) or (target_arr.ndim == logits_data.ndim - 1)
        if is_index_target:
            target_idx = target_arr.astype(int)
            if logits_data.ndim == 1:
                nll = -log_probs[target_idx.item()]
            else:
                nll = -log_probs[np.arange(logits_data.shape[0]), target_idx]
        else:
            nll = -np.sum(target_arr * log_probs, axis=-1)

        if self.reduction == "mean":
            loss_val = np.mean(nll)
        elif self.reduction == "sum":
            loss_val = np.sum(nll)
        else:
            loss_val = nll

        req_grad = is_grad_enabled() and logits.requires_grad
        out = Tensor(
            loss_val,
            requires_grad=req_grad,
            _children=(logits,) if req_grad else (),
            _op="CrossEntropyLoss",
            dtype=logits.dtype,
        )

        if req_grad:
            def _backward() -> None:
                if is_index_target:
                    target_idx = target_arr.astype(int)
                    d_logits = probs.copy()
                    if logits_data.ndim == 1:
                        d_logits[target_idx.item()] -= 1.0
                    else:
                        d_logits[np.arange(logits_data.shape[0]), target_idx] -= 1.0
                else:
                    d_logits = probs - target_arr

                if self.reduction == "mean":
                    scale = float(nll.size)
                    d_logits = d_logits * (out.grad / scale)
                elif self.reduction == "sum":
                    d_logits = d_logits * out.grad
                else:
                    d_logits = d_logits * out.grad[..., None]

                _accumulate_grad(logits, d_logits)

            out._backward = _backward

        return out
