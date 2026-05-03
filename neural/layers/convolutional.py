from __future__ import annotations

"""
Dilated convolution layers with parallel branches.

Provides MultiDilationConv{1,2,3}d: stacks multiple convolutions
with different dilation rates applied in parallel to the same input,
then concatenates the outputs along the channel dimension.

Useful for capture multi-scale context without increasing parameters.
"""

import torch
import torch.nn as nn


class MultiDilationConv1d(nn.Module):
    """Parallel dilated 1-D convolutions with channels split evenly across branches.

    Creates *k* nn.Conv1d layers (k = len(dilations)), each with a different
    dilation rate, then concatenates their outputs along the channel axis.
    This produces the same effective receptive-field expansion as a dilated
    convolutional block while keeping each branch shallow.

    Example::

        # Dual-dilation block: receptive fields of 3 and 5
        block = MultiDilationConv1d(64, 128, kernel_size=3, dilations=[1, 2])
        # out_channels = 128, out_channels_per_conv = 64

        # Three-way dilation
        block = MultiDilationConv1d(64, 96, kernel_size=5, dilations=[1, 2, 4])
        # out_channels_per_conv = 32
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | list[int] | tuple[int, ...] = 3,
        padding: str = "same",
        dilations: list[int] | tuple[int, ...] = (1,),
        bias: bool = True,
        stride: int = 1,
    ) -> None:
        super().__init__()

        n_branches = len(dilations)
        if out_channels % n_branches != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by "
                f"len(dilations) ({n_branches})"
            )

        if padding != "same":
            raise ValueError("Only padding='same' is supported.")

        if isinstance(kernel_size, int):
            kernel_sizes: list[int] = [kernel_size] * n_branches
        else:
            kernel_sizes = list(kernel_size)
        if len(kernel_sizes) != n_branches:
            raise ValueError(
                f"len(kernel_size) ({len(kernel_sizes)}) must equal "
                f"len(dilations) ({n_branches})"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_branches = n_branches
        self.out_channels_per_branch = out_channels // n_branches

        # Register one Conv1d per dilation.  nn.ModuleList ensures
        # parameters are discovered by .parameters() and .to(device).
        self.branches = nn.ModuleList()
        for ks, dilation in zip(kernel_sizes, dilations):
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=self.out_channels_per_branch,
                kernel_size=ks,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
            )
            self.branches.append(conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply parallel dilated convolutions and concatenate outputs."""
        outputs = [branch(x) for branch in self.branches]
        return torch.cat(outputs, dim=1)


class MultiDilationConv2d(nn.Module):
    """Parallel dilated 2-D convolutions with channels split evenly across branches.

    Same design as MultiDilationConv1d but for image data (N, C, H, W).

    Example::

        block = MultiDilationConv2d(
            in_channels=32, out_channels=64, kernel_size=3,
            dilations=[1, 2, 4]
        )
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | list[int] | tuple[int, ...] = 3,
        padding: str = "same",
        dilations: list[int] | tuple[int, ...] = (1,),
        bias: bool = True,
        stride: int = 1,
    ) -> None:
        super().__init__()

        n_branches = len(dilations)
        if out_channels % n_branches != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by "
                f"len(dilations) ({n_branches})"
            )

        if padding != "same":
            raise ValueError("Only padding='same' is supported.")

        if isinstance(kernel_size, int):
            kernel_sizes: list[int] = [kernel_size] * n_branches
        else:
            kernel_sizes = list(kernel_size)
        if len(kernel_sizes) != n_branches:
            raise ValueError(
                f"len(kernel_size) ({len(kernel_sizes)}) must equal "
                f"len(dilations) ({n_branches})"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_branches = n_branches
        self.out_channels_per_branch = out_channels // n_branches

        self.branches = nn.ModuleList()
        for ks, dilation in zip(kernel_sizes, dilations):
            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.out_channels_per_branch,
                kernel_size=ks,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
            )
            self.branches.append(conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply parallel dilated convolutions and concatenate outputs."""
        outputs = [branch(x) for branch in self.branches]
        return torch.cat(outputs, dim=1)


class MultiDilationConv3d(nn.Module):
    """Parallel dilated 3-D convolutions with channels split evenly across branches.

    Same design but for volumetric data (N, C, D, H, W) -- useful for
    medical imaging, video, or 3-D CNNs.

    Example::

        block = MultiDilationConv3d(
            in_channels=1, out_channels=32, kernel_size=3,
            dilations=[1, 2]
        )
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | list[int] | tuple[int, ...] = 3,
        padding: str = "same",
        dilations: list[int] | tuple[int, ...] = (1,),
        bias: bool = True,
        stride: int = 1,
    ) -> None:
        super().__init__()

        n_branches = len(dilations)
        if out_channels % n_branches != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by "
                f"len(dilations) ({n_branches})"
            )

        if padding != "same":
            raise ValueError("Only padding='same' is supported.")

        if isinstance(kernel_size, int):
            kernel_sizes: list[int] = [kernel_size] * n_branches
        else:
            kernel_sizes = list(kernel_size)
        if len(kernel_sizes) != n_branches:
            raise ValueError(
                f"len(kernel_size) ({len(kernel_sizes)}) must equal "
                f"len(dilations) ({n_branches})"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_branches = n_branches
        self.out_channels_per_branch = out_channels // n_branches

        self.branches = nn.ModuleList()
        for ks, dilation in zip(kernel_sizes, dilations):
            conv = nn.Conv3d(
                in_channels=in_channels,
                out_channels=self.out_channels_per_branch,
                kernel_size=ks,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
            )
            self.branches.append(conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply parallel dilated convolutions and concatenate outputs."""
        outputs = [branch(x) for branch in self.branches]
        return torch.cat(outputs, dim=1)


class SubPixelConv2d(nn.Module):
    """Sub-pixel upscaling convolution (pixel shuffle).

    Applies parallel convolutions with different kernel sizes and/or dilations,
    each producing ``scale_factor^2 * out_channels_per_branch`` channels, then
    rearranges them via pixel_shuffle into ``out_channels`` output channels
    with spatial dimensions scaled by ``scale_factor``.

    Example::

        # Upscale 2x, single kernel
        up = SubPixelConv2d(64, 32, scale_factor=2)
        # input: (N, 64, H, W) -> output: (N, 32, 2*H, 2*W)

        # Multi-branch with per-branch kernel sizes
        up = SubPixelConv2d(
            in_channels=64, out_channels=32, scale_factor=2,
            kernel_size=[3, 5], dilations=[1, 2]
        )
        # channels split across 2 branches -> 2x pixel_shuffle upscale
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        kernel_size: int | list[int] | tuple[int, ...] = 3,
        padding: str = "same",
        dilations: list[int] | tuple[int, ...] = (1,),
        bias: bool = True,
        stride: int = 1,
    ) -> None:
        super().__init__()

        n_branches = len(dilations)
        ps_sq = scale_factor ** 2

        if out_channels % n_branches != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by "
                f"len(dilations) ({n_branches})"
            )

        per_branch_channels = (out_channels // n_branches) * ps_sq
        if padding != "same":
            raise ValueError("Only padding='same' is supported.")

        if isinstance(kernel_size, int):
            kernel_sizes: list[int] = [kernel_size] * n_branches
        else:
            kernel_sizes = list(kernel_size)
        if len(kernel_sizes) != n_branches:
            raise ValueError(
                f"len(kernel_size) ({len(kernel_sizes)}) must equal "
                f"len(dilations) ({n_branches})"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_branches = n_branches
        self.out_channels_per_branch = out_channels // n_branches

        # Each branch produces scale_factor^2 * per_branch_channels before shuffle.
        self.branches = nn.ModuleList()
        for ks, dilation in zip(kernel_sizes, dilations):
            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=per_branch_channels,
                kernel_size=ks,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
            )
            self.branches.append(conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply parallel convolutions and pixel shuffle for each branch."""
        outputs = []
        for branch in self.branches:
            out = branch(x)  # (N, ps_sq * per_branch, H, W)
            out = torch.pixel_shuffle(out, self.scale_factor)  # (N, per_branch, sH, sW)
            outputs.append(out)
        return torch.cat(outputs, dim=1)  # (N, out_channels, sH, sW)
