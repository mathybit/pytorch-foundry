"""
Example regressor DNN — Model + build_model() wrapper.

build_model() returns a torch.nn.Module ready for training.
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from neural.builders import build_dnn_model


class RegressorDNN(nn.Module):
    """Regressor DNN using the shared DNN builder."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self._model = build_dnn_model(config)

    def forward(self, x: torch.Tensor, embedding: bool = False) -> torch.Tensor:
        return self._model(x, embedding=embedding)


def build_model(config: Dict[str, Any]) -> nn.Module:
    """Build and return the regressor_dnn model.

    Args:
        config: Dict with d_input, d_hidden, d_output, activation,
            batch_norm, bn_affine, bn_track_running_stats, dropout.

    Returns:
        nn.Module ready for training.
    """
    return RegressorDNN(config)
