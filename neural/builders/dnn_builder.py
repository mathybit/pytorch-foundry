"""
Generic DNN Model Builder

Builds feedforward DNNs with configurable:
- Hidden layer dimensions
- Per-layer or global activation functions
- Per-layer or global batch normalization
- Dropout
- Embedding extraction (returns pre-logits layer output)

All models return raw logits (no output activation).
"""

from typing import Any, Dict, List, Union

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


class DNNModel(nn.Module):
    """Generic DNN with configurable hidden layers.

    Architecture::

        Input -> [Linear -> BatchNorm1d -> Activation -> Dropout]* -> Linear(d_prev, d_output) -> Output

    Returns raw logits. Supports embedding extraction via ``forward(embedding=True)``,
    which returns the output of the last hidden layer (pre-logits).

    If ``d_hidden`` is empty/None, the model is a single ``Linear(d_input, d_output)``
    layer. In this case ``embedding=True`` returns the same as ``embedding=False``.

    Args:
        config: Dict with keys:
            d_input (int): input dimension
            d_hidden (list[int] or None): hidden layer sizes
            d_output (int): output dimension
            activation (str or list[str]): activation function(s)
            batch_norm (bool or list[bool]): batch norm flag(s)
            bn_affine (bool): BatchNorm1d affine parameter (default True)
            bn_track_running_stats (bool): BatchNorm1d track_running_stats parameter (default True)
            dropout (float): dropout rate applied after each hidden layer
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()

        d_hidden = config.get("d_hidden") or []
        activation_names = self._expand_to_list(config["activation"], len(d_hidden))
        bn_flags = self._expand_to_list(config["batch_norm"], len(d_hidden))

        if len(d_hidden) != len(activation_names):
            raise ValueError(
                f"activation list length ({len(activation_names)}) must match "
                f"d_hidden length ({len(d_hidden)})"
            )
        if len(d_hidden) != len(bn_flags):
            raise ValueError(
                f"batch_norm list length ({len(bn_flags)}) must match "
                f"d_hidden length ({len(d_hidden)})"
            )

        for act in activation_names:
            if act not in _ACTIVATIONS:
                raise ValueError(
                    f"Unknown activation {act!r}. "
                    f"Choose from: {', '.join(sorted(_ACTIVATIONS))}"
                )

        layers = []
        d_prev = config["d_input"]
        bn_affine = config.get("bn_affine", True)
        bn_track = config.get("bn_track_running_stats", True)
        dropout = config.get("dropout", 0.0)

        for d_curr, act_name, do_bn in zip(d_hidden, activation_names, bn_flags):
            layers.append(nn.Linear(d_prev, d_curr))
            if do_bn:
                layers.append(nn.BatchNorm1d(d_curr, affine=bn_affine, track_running_stats=bn_track))
            layers.append(_ACTIVATIONS[act_name]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d_prev = d_curr

        self.hidden = nn.Sequential(*layers) if d_hidden else nn.Identity()
        self.output = nn.Linear(d_prev, config["d_output"])
        self._has_hidden = len(d_hidden) > 0

    @staticmethod
    def _expand_to_list(value, length):
        """Expand single value to list of length."""
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value] * length

    @staticmethod
    def _make_activation(name):
        """Create activation layer from string name."""
        return _ACTIVATIONS[name]()

    def forward(self, x: torch.Tensor, embedding: bool = False) -> torch.Tensor:
        """Forward pass.

        Args:
            x: input tensor (..., d_input).
            embedding: if True, return pre-logits layer output.

        Returns:
            Logits (..., d_output) when embedding=False.
            Hidden representation (..., last_hidden_dim) when embedding=True
            and hidden layers exist.
            Logits (..., d_output) when embedding=True but no hidden layers.
        """
        h = self.hidden(x)
        if embedding and self._has_hidden:
            return h
        return self.output(h)


def build_dnn_model(config: Dict[str, Any]) -> nn.Module:
    """Build and return a generic DNN model from config dict.

    Args:
        config: Dict with d_input, d_hidden, d_output, activation,
            batch_norm, bn_affine, bn_track_running_stats, dropout.

    Returns:
        nn.Module (DNNModel) ready for training.
    """
    return DNNModel(config)
