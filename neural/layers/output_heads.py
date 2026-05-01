"""
Generic Output Heads

Domain-agnostic prediction heads that attach to any fixed-size embedding
vector. Each head consists of a hidden layer followed by an output layer.
Both logits and intermediate (hidden) representations are accessible.
"""

import torch
import torch.nn as nn


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "celu": nn.CELU,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "leakyrelu": nn.LeakyReLU,
    "softplus": nn.Softplus,
    "softsign": nn.Softsign,
}


class ClassificationHead(nn.Module):
    """Multi-class classification head.

    Architecture: ``Linear(d_input, d_hidden) -> activation -> dropout -> Linear(d_hidden, n_classes)``

    Returns raw logits for ``CrossEntropyLoss``.  Pass ``embedding=True``
    to the forward pass to retrieve the hidden layer representation instead.

    Example::

        head = ClassificationHead(512, 256, n_classes=10)
        logits = head(input)       # (B, 10)
        hidden = head(input, embedding=True)  # (B, 256)
    """

    def __init__(
        self,
        d_input: int,
        d_hidden: int,
        n_classes: int,
        activation: str = "relu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation {activation!r}. "
                f"Choose from: {', '.join(sorted(_ACTIVATIONS))}"
            )

        self.hidden = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            _ACTIVATIONS[activation](),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(d_hidden, n_classes)

    def forward(self, x: torch.Tensor, embedding: bool = False) -> torch.Tensor:
        """Forward pass.

        Args:
            x: input tensor (B, d_input).
            embedding: if True, return hidden representation (B, d_hidden).

        Returns:
            Logits (B, n_classes) when embedding=False;
            hidden representation (B, d_hidden) when embedding=True.
        """
        h = self.hidden(x)
        if embedding:
            return h
        return self.output(h)


class RegressionHead(nn.Module):
    """Regression head.

    Architecture: ``Linear(d_input, d_hidden) -> activation -> dropout -> Linear(d_hidden, 1)``

    Example::

        head = RegressionHead(512, 256)
        pred = head(input)  # (B, 1)
        hidden = head(input, embedding=True)  # (B, 256)
    """

    def __init__(
        self,
        d_input: int,
        d_hidden: int,
        activation: str = "relu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation {activation!r}. "
                f"Choose from: {', '.join(sorted(_ACTIVATIONS))}"
            )

        self.hidden = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            _ACTIVATIONS[activation](),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(d_hidden, 1)

    def forward(self, x: torch.Tensor, embedding: bool = False) -> torch.Tensor:
        """Forward pass.

        Args:
            x: input tensor (B, d_input).
            embedding: if True, return hidden representation (B, d_hidden).

        Returns:
            Prediction (B, 1) when embedding=False;
            hidden representation (B, d_hidden) when embedding=True.
        """
        h = self.hidden(x)
        if embedding:
            return h
        return self.output(h)
