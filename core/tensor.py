"""
core/tensor.py - Multi-dimensional array node in a dynamic computation graph.
"""

from typing import Union, Tuple, Set, Optional, Sequence, Any
import numpy as np

from core.autograd import is_grad_enabled, unbroadcast, backward as run_backward


def _accumulate_grad(tensor: "Tensor", delta: np.ndarray) -> None:
    """Safely accumulate gradient into a tensor respecting requires_grad and dtype."""
    if not tensor.requires_grad:
        return
    delta_arr = np.asarray(delta, dtype=tensor.dtype)
    if tensor.grad is None:
        tensor.grad = np.array(delta_arr, dtype=tensor.dtype, copy=True)
    else:
        tensor.grad += delta_arr


class Tensor:
    """
    A multi-dimensional array node in a dynamic computation graph.
    Supports reverse-mode automatic differentiation with NumPy backend.
    """

    def __init__(
        self,
        data: Union[int, float, list, np.ndarray, "Tensor"],
        requires_grad: bool = False,
        _children: Tuple["Tensor", ...] = (),
        _op: str = "",
        dtype: Optional[Union[np.dtype, type, str]] = None,
    ) -> None:
        if isinstance(data, Tensor):
            data = data.data

        if dtype is not None:
            target_dtype = np.dtype(dtype)
            if isinstance(data, np.ndarray):
                self.data = data.astype(target_dtype) if data.dtype != target_dtype else data
            else:
                self.data = np.array(data, dtype=target_dtype)
        else:
            if isinstance(data, np.ndarray):
                # Preserve float64 if explicitly given (e.g. for high-precision grad checks)
                if data.dtype == np.float64:
                    self.data = data
                else:
                    self.data = data.astype(np.float32) if data.dtype != np.float32 else data
            else:
                self.data = np.array(data, dtype=np.float32)

        # Gradient tracking flag
        self.requires_grad: bool = requires_grad

        # Gradient storage (deferred until backward())
        self.grad: Optional[np.ndarray] = None

        # DAG ancestors that produced this node
        self._prev: Set["Tensor"] = set(_children)

        # Local chain-rule backward closure
        self._backward = lambda: None

        # Operation tag for debugging/graph visualization
        self._op: str = _op

    # -------------------------------------------------------------------------
    # Properties & Utilities
    # -------------------------------------------------------------------------

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def numpy(self) -> np.ndarray:
        return self.data

    def item(self) -> Union[int, float]:
        return self.data.item()

    def tolist(self) -> list:
        return self.data.tolist()

    def zero_grad(self) -> None:
        """Reset gradient to None."""
        self.grad = None

    def detach(self) -> "Tensor":
        """Return a new Tensor detached from the current computation graph."""
        return Tensor(self.data.copy(), requires_grad=False, dtype=self.dtype)

    def backward(self, grad: Optional[np.ndarray] = None) -> None:
        """Run backpropagation from this root tensor."""
        run_backward(self, grad=grad)

    def __repr__(self) -> str:
        grad_fn = f", grad_fn=<{self._op}>" if self._op else ""
        return f"Tensor({self.data}, requires_grad={self.requires_grad}{grad_fn})"

    # -------------------------------------------------------------------------
    # Arithmetic Operations
    # -------------------------------------------------------------------------

    def __add__(self, other: Any) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other, dtype=self.dtype)
        req_grad = is_grad_enabled() and (self.requires_grad or other.requires_grad)

        out = Tensor(
            self.data + other.data,
            requires_grad=req_grad,
            _children=(self, other) if req_grad else (),
            _op="+",
            dtype=self.dtype,
        )

        if req_grad:
            def _backward() -> None:
                _accumulate_grad(self, unbroadcast(out.grad, self.shape))
                _accumulate_grad(other, unbroadcast(out.grad, other.shape))
            out._backward = _backward

        return out

    def __radd__(self, other: Any) -> "Tensor":
        return self.__add__(other)

    def __neg__(self) -> "Tensor":
        req_grad = is_grad_enabled() and self.requires_grad
        out = Tensor(
            -self.data,
            requires_grad=req_grad,
            _children=(self,) if req_grad else (),
            _op="neg",
            dtype=self.dtype,
        )

        if req_grad:
            def _backward() -> None:
                _accumulate_grad(self, -out.grad)
            out._backward = _backward

        return out

    def __sub__(self, other: Any) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other, dtype=self.dtype)
        req_grad = is_grad_enabled() and (self.requires_grad or other.requires_grad)

        out = Tensor(
            self.data - other.data,
            requires_grad=req_grad,
            _children=(self, other) if req_grad else (),
            _op="-",
            dtype=self.dtype,
        )

        if req_grad:
            def _backward() -> None:
                _accumulate_grad(self, unbroadcast(out.grad, self.shape))
                _accumulate_grad(other, unbroadcast(-out.grad, other.shape))
            out._backward = _backward

        return out

    def __rsub__(self, other: Any) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other, dtype=self.dtype)
        return other.__sub__(self)

    def __mul__(self, other: Any) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other, dtype=self.dtype)
        req_grad = is_grad_enabled() and (self.requires_grad or other.requires_grad)

        out = Tensor(
            self.data * other.data,
            requires_grad=req_grad,
            _children=(self, other) if req_grad else (),
            _op="*",
            dtype=self.dtype,
        )

        if req_grad:
            def _backward() -> None:
                _accumulate_grad(self, unbroadcast(out.grad * other.data, self.shape))
                _accumulate_grad(other, unbroadcast(out.grad * self.data, other.shape))
            out._backward = _backward

        return out

    def __rmul__(self, other: Any) -> "Tensor":
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other, dtype=self.dtype)
        req_grad = is_grad_enabled() and (self.requires_grad or other.requires_grad)

        out = Tensor(
            self.data / other.data,
            requires_grad=req_grad,
            _children=(self, other) if req_grad else (),
            _op="/",
            dtype=self.dtype,
        )

        if req_grad:
            def _backward() -> None:
                _accumulate_grad(self, unbroadcast(out.grad / other.data, self.shape))
                _accumulate_grad(
                    other,
                    unbroadcast(-out.grad * self.data / (other.data ** 2), other.shape),
                )
            out._backward = _backward

        return out

    def __rtruediv__(self, other: Any) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other, dtype=self.dtype)
        return other.__truediv__(self)

    def __pow__(self, power: Union[int, float, "Tensor"]) -> "Tensor":
        is_tensor_pow = isinstance(power, Tensor)
        power_t = power if is_tensor_pow else Tensor(power, dtype=self.dtype)
        req_grad = is_grad_enabled() and (self.requires_grad or power_t.requires_grad)

        out = Tensor(
            self.data ** power_t.data,
            requires_grad=req_grad,
            _children=(self, power_t) if req_grad else (),
            _op="**",
            dtype=self.dtype,
        )

        if req_grad:
            def _backward() -> None:
                # dz/dx = dz * p * x^(p-1)
                gx = out.grad * power_t.data * (self.data ** (power_t.data - 1))
                _accumulate_grad(self, unbroadcast(gx, self.shape))
                if is_tensor_pow and power_t.requires_grad:
                    # dz/dp = dz * (x^p) * ln(x)
                    safe_x = np.where(self.data > 0, self.data, 1.0)
                    gp = out.grad * (self.data ** power_t.data) * np.log(safe_x)
                    _accumulate_grad(power_t, unbroadcast(gp, power_t.shape))
            out._backward = _backward

        return out

    def __matmul__(self, other: Any) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other, dtype=self.dtype)
        req_grad = is_grad_enabled() and (self.requires_grad or other.requires_grad)

        out = Tensor(
            self.data @ other.data,
            requires_grad=req_grad,
            _children=(self, other) if req_grad else (),
            _op="@",
            dtype=self.dtype,
        )

        if req_grad:
            def _backward() -> None:
                # Gradient w.r.t self (x)
                if self.requires_grad:
                    if self.ndim == 1 and other.ndim == 1:
                        gx = out.grad * other.data
                    elif self.ndim == 1:
                        # self is (K,), other is (..., K, N) -> out.grad is (..., N)
                        g = out.grad[..., None, :]
                        gx = (g @ np.swapaxes(other.data, -1, -2))[..., 0, :]
                    elif other.ndim == 1:
                        # self is (..., M, K), other is (K,) -> out.grad is (..., M)
                        g = out.grad[..., :, None]
                        gx = g @ other.data[None, :]
                    else:
                        gx = out.grad @ np.swapaxes(other.data, -1, -2)
                    _accumulate_grad(self, unbroadcast(gx, self.shape))

                # Gradient w.r.t other (y)
                if other.requires_grad:
                    if self.ndim == 1 and other.ndim == 1:
                        gy = out.grad * self.data
                    elif self.ndim == 1:
                        gy = self.data[..., :, None] @ out.grad[..., None, :]
                    elif other.ndim == 1:
                        g = out.grad[..., :, None]
                        gy = (np.swapaxes(self.data, -1, -2) @ g)[..., :, 0]
                    else:
                        gy = np.swapaxes(self.data, -1, -2) @ out.grad
                    _accumulate_grad(other, unbroadcast(gy, other.shape))

            out._backward = _backward

        return out

    def __rmatmul__(self, other: Any) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other, dtype=self.dtype)
        return other.__matmul__(self)

    # -------------------------------------------------------------------------
    # Reductions & Reshaping
    # -------------------------------------------------------------------------

    def sum(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> "Tensor":
        req_grad = is_grad_enabled() and self.requires_grad
        out_data = self.data.sum(axis=axis, keepdims=keepdims)

        out = Tensor(
            out_data,
            requires_grad=req_grad,
            _children=(self,) if req_grad else (),
            _op="sum",
            dtype=self.dtype,
        )

        if req_grad:
            def _backward() -> None:
                grad = out.grad
                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                gx = np.broadcast_to(grad, self.shape)
                _accumulate_grad(self, gx)
            out._backward = _backward

        return out

    def mean(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> "Tensor":
        req_grad = is_grad_enabled() and self.requires_grad
        out_data = self.data.mean(axis=axis, keepdims=keepdims)

        out = Tensor(
            out_data,
            requires_grad=req_grad,
            _children=(self,) if req_grad else (),
            _op="mean",
            dtype=self.dtype,
        )

        if req_grad:
            def _backward() -> None:
                grad = out.grad
                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                scale = float(self.data.size) / float(out_data.size)
                gx = np.broadcast_to(grad / scale, self.shape)
                _accumulate_grad(self, gx)
            out._backward = _backward

        return out

    def reshape(self, *shape: Any) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            target_shape = tuple(shape[0])
        else:
            target_shape = tuple(shape)

        req_grad = is_grad_enabled() and self.requires_grad
        out = Tensor(
            self.data.reshape(target_shape),
            requires_grad=req_grad,
            _children=(self,) if req_grad else (),
            _op="reshape",
            dtype=self.dtype,
        )

        if req_grad:
            def _backward() -> None:
                _accumulate_grad(self, out.grad.reshape(self.shape))
            out._backward = _backward

        return out

    def transpose(self, *axes: Any) -> "Tensor":
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            target_axes: Optional[Tuple[int, ...]] = tuple(axes[0])
        elif len(axes) > 1:
            target_axes = tuple(axes)
        else:
            target_axes = None

        req_grad = is_grad_enabled() and self.requires_grad
        out_data = self.data.transpose(target_axes) if target_axes is not None else self.data.transpose()

        out = Tensor(
            out_data,
            requires_grad=req_grad,
            _children=(self,) if req_grad else (),
            _op="transpose",
            dtype=self.dtype,
        )

        if req_grad:
            def _backward() -> None:
                if target_axes is None:
                    _accumulate_grad(self, out.grad.transpose())
                else:
                    inv_axes = tuple(np.argsort(target_axes))
                    _accumulate_grad(self, out.grad.transpose(inv_axes))
            out._backward = _backward

        return out

    def squeeze(self, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> "Tensor":
        req_grad = is_grad_enabled() and self.requires_grad
        out = Tensor(
            self.data.squeeze(axis=axis),
            requires_grad=req_grad,
            _children=(self,) if req_grad else (),
            _op="squeeze",
            dtype=self.dtype,
        )

        if req_grad:
            def _backward() -> None:
                _accumulate_grad(self, out.grad.reshape(self.shape))
            out._backward = _backward

        return out

    def flatten(self) -> "Tensor":
        return self.reshape(-1)

    # -------------------------------------------------------------------------
    # Elementwise Math & Activations
    # -------------------------------------------------------------------------

    def relu(self) -> "Tensor":
        req_grad = is_grad_enabled() and self.requires_grad
        out = Tensor(
            np.maximum(self.data, 0),
            requires_grad=req_grad,
            _children=(self,) if req_grad else (),
            _op="relu",
            dtype=self.dtype,
        )

        if req_grad:
            def _backward() -> None:
                _accumulate_grad(self, out.grad * (self.data > 0))
            out._backward = _backward

        return out

    def sigmoid(self) -> "Tensor":
        req_grad = is_grad_enabled() and self.requires_grad
        # Numerically clipped sigmoid to prevent overflow
        clipped = np.clip(self.data, -500.0, 500.0)
        s = 1.0 / (1.0 + np.exp(-clipped))

        out = Tensor(
            s,
            requires_grad=req_grad,
            _children=(self,) if req_grad else (),
            _op="sigmoid",
            dtype=self.dtype,
        )

        if req_grad:
            def _backward() -> None:
                _accumulate_grad(self, out.grad * s * (1.0 - s))
            out._backward = _backward

        return out

    def tanh(self) -> "Tensor":
        req_grad = is_grad_enabled() and self.requires_grad
        t = np.tanh(self.data)

        out = Tensor(
            t,
            requires_grad=req_grad,
            _children=(self,) if req_grad else (),
            _op="tanh",
            dtype=self.dtype,
        )

        if req_grad:
            def _backward() -> None:
                _accumulate_grad(self, out.grad * (1.0 - t ** 2))
            out._backward = _backward

        return out

    def exp(self) -> "Tensor":
        req_grad = is_grad_enabled() and self.requires_grad
        e = np.exp(np.clip(self.data, -500.0, 500.0))

        out = Tensor(
            e,
            requires_grad=req_grad,
            _children=(self,) if req_grad else (),
            _op="exp",
            dtype=self.dtype,
        )

        if req_grad:
            def _backward() -> None:
                _accumulate_grad(self, out.grad * e)
            out._backward = _backward

        return out

    def log(self) -> "Tensor":
        req_grad = is_grad_enabled() and self.requires_grad
        # Safe log with floor epsilon
        safe_data = np.maximum(self.data, 1e-15)
        out = Tensor(
            np.log(safe_data),
            requires_grad=req_grad,
            _children=(self,) if req_grad else (),
            _op="log",
            dtype=self.dtype,
        )

        if req_grad:
            def _backward() -> None:
                _accumulate_grad(self, out.grad / safe_data)
            out._backward = _backward

        return out

    def maximum(self, other: Any) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other, dtype=self.dtype)
        req_grad = is_grad_enabled() and (self.requires_grad or other.requires_grad)

        out = Tensor(
            np.maximum(self.data, other.data),
            requires_grad=req_grad,
            _children=(self, other) if req_grad else (),
            _op="maximum",
            dtype=self.dtype,
        )

        if req_grad:
            def _backward() -> None:
                mask_self = self.data >= other.data
                mask_other = other.data > self.data
                _accumulate_grad(self, unbroadcast(out.grad * mask_self, self.shape))
                _accumulate_grad(other, unbroadcast(out.grad * mask_other, other.shape))
            out._backward = _backward

        return out

    def clip(self, min_val: Optional[float] = None, max_val: Optional[float] = None) -> "Tensor":
        req_grad = is_grad_enabled() and self.requires_grad
        out_data = np.clip(
            self.data,
            a_min=-np.inf if min_val is None else min_val,
            a_max=np.inf if max_val is None else max_val,
        )

        out = Tensor(
            out_data,
            requires_grad=req_grad,
            _children=(self,) if req_grad else (),
            _op="clip",
            dtype=self.dtype,
        )

        if req_grad:
            def _backward() -> None:
                mask = np.ones_like(self.data, dtype=bool)
                if min_val is not None:
                    mask &= (self.data >= min_val)
                if max_val is not None:
                    mask &= (self.data <= max_val)
                _accumulate_grad(self, out.grad * mask)
            out._backward = _backward

        return out

    def softmax(self, axis: int = -1) -> "Tensor":
        req_grad = is_grad_enabled() and self.requires_grad
        m = np.max(self.data, axis=axis, keepdims=True)
        exp_data = np.exp(self.data - m)
        sum_exp = np.sum(exp_data, axis=axis, keepdims=True)
        probs = exp_data / sum_exp

        out = Tensor(
            probs,
            requires_grad=req_grad,
            _children=(self,) if req_grad else (),
            _op="softmax",
            dtype=self.dtype,
        )

        if req_grad:
            def _backward() -> None:
                dot = np.sum(out.grad * probs, axis=axis, keepdims=True)
                gx = probs * (out.grad - dot)
                _accumulate_grad(self, gx)
            out._backward = _backward

        return out
