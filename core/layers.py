"""
core/layers.py - Neural network layers, parameter management, and activations.
"""

from typing import List, Dict, Tuple, Optional, Any, Iterator, Union
import numpy as np

from core.tensor import Tensor


class Parameter(Tensor):
    """
    A trainable Tensor parameter inside a Module.
    Always initializes with requires_grad=True.
    """

    def __init__(
        self,
        data: Union[int, float, list, np.ndarray, Tensor],
        dtype: Optional[Union[np.dtype, type, str]] = None,
    ) -> None:
        super().__init__(data, requires_grad=True, dtype=dtype)


class Module:
    """
    Base class for all neural network modules.
    Tracks parameters, submodules, and training/evaluation state.
    """

    def __init__(self) -> None:
        self._parameters: Dict[str, Parameter] = {}
        self._modules: Dict[str, "Module"] = {}
        self.training: bool = True

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Parameter):
            if not hasattr(self, "_parameters"):
                super().__setattr__("_parameters", {})
            self._parameters[name] = value
        elif isinstance(value, Module):
            if not hasattr(self, "_modules"):
                super().__setattr__("_modules", {})
            self._modules[name] = value

        super().__setattr__(name, value)

    def parameters(self) -> List[Parameter]:
        """Return a list of all parameters in this module and its submodules."""
        params: List[Parameter] = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    def zero_grad(self) -> None:
        """Zero out gradients for all tracked parameters."""
        for p in self.parameters():
            p.zero_grad()

    def train(self, mode: bool = True) -> "Module":
        """Set module and submodules into training mode."""
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self) -> "Module":
        """Set module and submodules into evaluation mode."""
        return self.train(False)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Subclasses of Module must implement forward().")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)


class Linear(Module):
    """
    Fully-connected linear transformation: y = x @ W + b.
    Weights are initialized using standard Kaiming uniform initialization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: Optional[Union[np.dtype, type, str]] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Kaiming uniform initialization: bound = 1 / sqrt(in_features)
        bound = 1.0 / np.sqrt(in_features) if in_features > 0 else 1.0
        w_data = np.random.uniform(-bound, bound, size=(in_features, out_features))
        self.weight = Parameter(w_data, dtype=dtype)

        if bias:
            b_data = np.random.uniform(-bound, bound, size=(out_features,))
            self.bias: Optional[Parameter] = Parameter(b_data, dtype=dtype)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class ReLU(Module):
    """Rectified Linear Unit activation: max(0, x)."""

    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Sigmoid(Module):
    """Sigmoid activation: 1 / (1 + exp(-x))."""

    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Tanh(Module):
    """Hyperbolic tangent activation: tanh(x)."""

    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class Softmax(Module):
    """Softmax activation along specified axis."""

    def __init__(self, axis: int = -1) -> None:
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        return x.softmax(axis=self.axis)


class Sequential(Module):
    """Sequential container passing input through a sequence of modules in order."""

    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.layers: List[Module] = []
        for i, mod in enumerate(modules):
            self.layers.append(mod)
            setattr(self, f"layer_{i}", mod)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, idx: int) -> Module:
        return self.layers[idx]

    def __iter__(self) -> Iterator[Module]:
        return iter(self.layers)
