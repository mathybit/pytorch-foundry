"""
Residual building blocks for convolutional networks.

Provides parameterized residual blocks that work with any ``nn.ConvNd``
class (``Conv1d``, ``Conv2d``, ``Conv3d``).

Each block takes ``conv=nn.ConvNd`` (default ``nn.Conv2d``) as argument
so it can be used in 1D, 2D, or 3D networks without code duplication.
"""

from typing import Literal

import torch
import torch.nn as nn


class SimpleResidualBlock(nn.Module):
    """Two convolutions with identity skip, no normalization.

    ``x -> conv -> activation -> conv -> + x``

    Useful as a lightweight residual unit when the surrounding layers
    (e.g. instance norm in a GAN) already handle normalization.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        conv: type[nn.Conv1d] | type[nn.Conv2d] | type[nn.Conv3d] = nn.Conv2d,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = conv(channels, channels, kernel_size, padding="same")
        self.conv2 = conv(channels, channels, kernel_size, padding="same")
        if isinstance(activation, str):
            act_map = {
                "relu": nn.ReLU,
                "prelu": lambda: nn.PReLU(num_parameters=channels),
                "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.2),
            }
            self.activation = act_map[activation]() if activation in act_map else None
        else:
            self.activation = activation
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        if self.activation is not None:
            out = self.activation(out)
        out = self.conv2(out)
        return out + residual


class ResidualBlock(nn.Module):
    """SRGAN-style residual block.

    ``x -> conv -> norm -> activation -> conv -> norm -> + x``

    Normalization is applied **before** activation (pre-activation / ResNet-V2)
    which is more stable for GAN training.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        conv: type[nn.Conv1d] | type[nn.Conv2d] | type[nn.Conv3d] = nn.Conv2d,
        norm: str = "batch",
        activation: Literal["relu", "prelu", "leaky_relu"] = "prelu",
        inplace: bool = False,
    ) -> None:
        super().__init__()

        norm_classes = {
            "batch": {"1d": nn.BatchNorm1d, "2d": nn.BatchNorm2d, "3d": nn.BatchNorm3d},
            "instance": {"1d": nn.InstanceNorm1d, "2d": nn.InstanceNorm2d, "3d": nn.InstanceNorm3d},
            "none": None,
        }

        conv_nd = conv(1, 1, 3)
        ndim = type(conv_nd).__name__.replace("Conv", "")
        ndim_lower = ndim.lower()

        if norm == "batch":
            norm_class = norm_classes["batch"][ndim_lower]
            self.norm1 = norm_class(channels)
            self.norm2 = norm_class(channels)
        elif norm == "instance":
            norm_class = norm_classes["instance"][ndim_lower]
            self.norm1 = norm_class(channels)
            self.norm2 = norm_class(channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        activation_classes = {
            "relu": lambda: nn.ReLU(inplace=inplace),
            "prelu": lambda: nn.PReLU(num_parameters=channels),
            "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.2, inplace=inplace),
        }
        self.activation = activation_classes[activation]()

        self.conv1 = conv(channels, channels, kernel_size, padding="same")
        self.conv2 = conv(channels, channels, kernel_size, padding="same")
        self.channels = channels
        self.norm_type = norm
        self.activation_name = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm1(x)
        out = self.activation(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.activation(out)
        out = self.conv2(out)
        return out + residual


class BottleneckResidualBlock(nn.Module):
    """Bottleneck residual block with 1x1 reduction/expansion.

    ``x -> conv(1x1, C -> C/r) -> norm -> act -> conv(kxk, C/r -> C/r) -> norm -> act
         -> conv(1x1, C/r -> C) -> + x``

    Reduces parameter count and computation for deep networks while
    maintaining similar accuracy.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        conv: type[nn.Conv1d] | type[nn.Conv2d] | type[nn.Conv3d] = nn.Conv2d,
        norm: str = "batch",
        activation: Literal["relu", "prelu", "leaky_relu"] = "relu",
        reduction: int = 4,
    ) -> None:
        super().__init__()

        bottleneck_channels = channels // reduction

        # Choose normalization
        conv_nd = conv(1, 1, 3)
        ndim = type(conv_nd).__name__.replace("Conv", "").lower()
        norm_map = {
            "batch": {"1d": nn.BatchNorm1d, "2d": nn.BatchNorm2d, "3d": nn.BatchNorm3d},
            "instance": {"1d": nn.InstanceNorm1d, "2d": nn.InstanceNorm2d, "3d": nn.InstanceNorm3d},
        }
        if norm in norm_map:
            norm_class = norm_map[norm][ndim]
            self.norm1 = norm_class(bottleneck_channels)
            self.norm2 = norm_class(bottleneck_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        # TODO: instead of creating instances here, use if/else logic
        act_map = {
            "relu": lambda: nn.ReLU(),
            "prelu": lambda: nn.PReLU(num_parameters=bottleneck_channels),
            "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.2),
        }
        self.activation = act_map[activation]()

        self.conv1 = conv(channels, bottleneck_channels, kernel_size=1, padding="same")
        self.conv2 = conv(bottleneck_channels, bottleneck_channels, kernel_size, padding="same")
        self.conv3 = conv(bottleneck_channels, channels, kernel_size=1, padding="same")
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation(out)

        out = self.conv3(out)
        return out + residual


class ResidualStack(nn.Module):
    """Chain N identical residual blocks with an optional input projection.

    ResNet style: a 1x1 conv projects the input to the stack width, then
    all blocks operate at that width.

    Example::

        # 3 SRGAN-style blocks, 64 channels throughout
        stack = ResidualStack(64, n_blocks=3, conv=nn.Conv2d)
        out = stack(x)  # (N, 64, H, W)

        # Input has 128 channels -> projected to 64 -> 3 residual blocks
        stack = ResidualStack(residual_channels=64, in_channels=128, conv=nn.Conv2d)
        out = stack(x)  # (N, 64, H, W)
    """

    def __init__(
        self,
        residual_channels: int,
        n_blocks: int = 3,
        block_type: Literal["simple", "srgan", "bottleneck"] = "srgan",
        in_channels: int | None = None,
        kernel_size: int = 3,
        conv: type[nn.Conv1d] | type[nn.Conv2d] | type[nn.Conv3d] = nn.Conv2d,
        norm: str = "batch",
        activation: str = "prelu",
        **block_kwargs,
    ) -> None:
        super().__init__()

        block_classes = {
            "simple": SimpleResidualBlock,
            "srgan": ResidualBlock,
            "preact": ResidualBlock,
            "bottleneck": BottleneckResidualBlock,
        }
        BlockClass = block_classes[block_type]

        # Project input to residual channel width when dimensions differ (ResNet style).
        self.projection: nn.Module | None = None
        if in_channels is not None and in_channels != residual_channels:
            self.projection = nn.Sequential(
                conv(in_channels, residual_channels, kernel_size=1, padding="same"),
            )

        blocks = []
        for _ in range(n_blocks):
            block = BlockClass(
                channels=residual_channels,
                kernel_size=kernel_size,
                conv=conv,
                norm=norm,
                activation=activation,
                **block_kwargs,
            )
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)
        self.residual_channels = residual_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.projection is not None:
            x = self.projection(x)
        for block in self.blocks:
            x = block(x)
        return x
