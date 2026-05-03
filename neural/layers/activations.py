"""
Activation layers for neural networks.

Provides Scaling, Mish, Swish, GELU, GLU, PReLU, and SEBlock —
custom, composable activations commonly needed in transformer and
attention-based architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Scaling(nn.Module):
    """Learnable scaling: y = scale * x.

    Useful for rescaling features after normalization or as a
    learnable gate on residual connections.

    Example::

        scale = Scaling(channels=64)
        # scale.weight: (64,) initialized to ones
        out = scale(x)  # (N, C, H, W) * (64, 1, 1) -> (N, C, H, W)
    """

    def __init__(self, channels: int = 1, init: float = 1.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full((channels,), init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = [1] * x.dim()
        shape[1] = self.scale.size(0)
        return x * self.scale.view(shape)


class Mish(nn.Module):
    """Mish activation: x * tanh(softplus(x)).

    Smooth non-monotonic activation that self-regularizes.
    Often outperforms ReLU in deep networks.

    Example::

        act = Mish()
        out = act(x)  # element-wise
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


class Swish(nn.Module):
    """Swish activation: x * sigmoid(beta * x).

    Generalized SiLU with a learnable slope parameter beta.
    When beta=1, this is equivalent to SiLU.

    Example::

        # Learnable beta (default)
        swish = Swish(trainable=True)

        # Fixed beta (SiLU)
        silu = Swish(trainable=False, beta=1.0)
    """

    def __init__(self, trainable: bool = True, beta: float = 1.0) -> None:
        super().__init__()
        if trainable:
            self.beta = nn.Parameter(torch.tensor(beta))
        else:
            self.register_buffer("beta", torch.tensor(beta))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)


class GELU(nn.Module):
    """GELU activation with tanh approximation.

    GELU(x) = x * Phi(x) where Phi is the standard normal CDF.
    Approximated as: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    Example::

        act = GELU()
        out = act(x)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi))
            * (x + 0.044715 * torch.pow(x, 3))
        ))


class GLU(nn.Module):
    """Gated Linear Unit.

    Splits the input along the channel dimension and applies:
    y = first_half * sigmoid(second_half)

    If dim=0 (batch), expects even first dim.
    If dim=1 (channels), expects even C.

    Example::

        glu = GLU(dim=1)
        # x: (N, 2*C, H, W) -> out: (N, C, H, W)
        out = glu(x)
    """

    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.glu(x, dim=self.dim)


class PReLU(nn.Module):
    """Parametric ReLU: max(x, alpha * x) + min(0, alpha * x).

    Also written as: max(x, 0) + alpha * min(x, 0).
    For negative inputs, outputs alpha * x (negative slope).

    Example::

        # Shared alpha (same slope for all channels)
        prelu = PReLU(num_params=1)

        # Per-channel alpha
        prelu = PReLU(num_params=64)
    """

    def __init__(self, num_params: int = 1, init: float = 0.25) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.full((num_params,), init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape weight for broadcasting on channel dim (dim=1)
        shape = [1] * x.dim()
        shape[1] = self.weight.size(0)
        w = self.weight.view(shape)
        pos = torch.clamp(x, min=0.0)
        neg = torch.clamp(x, max=0.0)
        return pos + w * neg


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention block.

    Squeezes global spatial information into a channel descriptor,
    then excites informative channels via two fully-connected layers
    with a ReLU intermediate and sigmoid output.

    Example::

        se = SEBlock(channels=64, reduction=16)
        # x: (N, 64, H, W) -> out: (N, 64, H, W)
        # output: x * attention_weights
        out = se(x)
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c = x.size(0), x.size(1)
        squeeze = x.mean(dim=[2, 3])            # (N, C)
        excite = self.fc1(squeeze)                # (N, mid)
        excite = self.relu(excite)                 # (N, mid)
        excite = self.fc2(excite)                 # (N, C)
        excite = self.sigmoid(excite)              # (N, C)
        scale = excite.view(n, c, 1, 1)           # (N, C, 1, 1)
        return x * scale


class SiLU(nn.Module):
    """SiLU activation: x * sigmoid(x).

    Also known as Swish-1. Same as Swish with beta=1.

    Example::

        act = SiLU()
        out = act(x)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class Activation(nn.Module):
    """Factory activation layer.

    Dispatches to a named activation type. Case-insensitive.

    Example::

        act = Activation("mish")
        act = Activation("swish", trainable=False, beta=1.0)
        act = Activation("glu", dim=1)
        act = Activation("prelu", num_params=64)
        act = Activation("leakyrelu", alpha=0.01)

    Type -> kwargs mapping:

    | type       | class       | extra kwargs     |
    |------------|-------------|------------------|
    | relu       | ReLU        | (none)           |
    | leakyrelu  | LeakyReLU   | alpha            |
    | elu        | ELU         | alpha            |
    | celu       | CELU        | alpha            |
    | mish       | Mish        | (none)           |
    | swish      | Swish       | trainable, beta  |
    | silu       | SiLU        | (none)           |
    | gelu       | GELU        | (none)           |
    | glu        | GLU         | dim              |
    | prelu      | PReLU       | num_params, init |
    | tanh       | Tanh        | (none)           |
    | sigmoid    | Sigmoid     | (none)           |
    | hardtanh   | Hardtanh    | min_val, max_val |
    | hardswish  | Hardswish   | (none)           |
    """

    _KWARGS = {
        "relu": set(),
        "leakyrelu": {"alpha"},
        "elu": {"alpha"},
        "celu": {"alpha"},
        "mish": set(),
        "swish": {"trainable", "beta"},
        "silu": set(),
        "gelu": set(),
        "glu": {"dim"},
        "prelu": {"num_params", "init"},
        "tanh": set(),
        "sigmoid": set(),
        "hardtanh": {"min_val", "max_val"},
        "hardswish": set(),
    }

    def __init__(self, type: str = "relu", **kwargs) -> None:  # noqa: A002
        super().__init__()
        t = type.lower()

        if t not in self._KWARGS:
            raise ValueError(
                f"Unknown activation type: {type!r}. "
                f"Choose from: {', '.join(sorted(self._KWARGS))}"
            )

        expected = self._KWARGS[t]
        extra = set(kwargs) - expected
        if extra:
            raise TypeError(
                f"{t}() got unexpected keyword argument(s): {', '.join(sorted(extra))}"
            )

        if t == "relu":
            self.act = nn.ReLU()
        elif t == "leakyrelu":
            alpha = kwargs.get("alpha", 0.01)
            self.act = nn.LeakyReLU(negative_slope=alpha)
        elif t == "elu":
            alpha = kwargs.get("alpha", 1.0)
            self.act = nn.ELU(alpha=alpha)
        elif t == "celu":
            alpha = kwargs.get("alpha", 1.0)
            self.act = nn.CELU(alpha=alpha)
        elif t == "mish":
            self.act = Mish()
        elif t == "swish":
            trainable = kwargs.get("trainable", True)
            beta = kwargs.get("beta", 1.0)
            self.act = Swish(trainable=trainable, beta=beta)
        elif t == "silu":
            self.act = SiLU()
        elif t == "gelu":
            self.act = GELU()
        elif t == "glu":
            dim = kwargs.get("dim", 1)
            self.act = GLU(dim=dim)
        elif t == "prelu":
            num_params = kwargs.get("num_params", 1)
            init = kwargs.get("init", 0.25)
            self.act = PReLU(num_params=num_params, init=init)
        elif t == "tanh":
            self.act = nn.Tanh()
        elif t == "sigmoid":
            self.act = nn.Sigmoid()
        elif t == "hardtanh":
            min_val = kwargs.get("min_val", -1.0)
            max_val = kwargs.get("max_val", 1.0)
            self.act = nn.Hardtanh(min_val=min_val, max_val=max_val)
        elif t == "hardswish":
            self.act = nn.Hardswish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)
