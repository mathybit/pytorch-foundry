"""
Normalization layers for neural networks.

Provides pixel normalization, adaptive instance normalization, and
a stackable normalization composition utility.
"""

import torch
import torch.nn as nn


class PixelNorm(nn.Module):
    """Pixel-wise L2 normalization.

    Divides each pixel by the L2 norm over channels.  Stabilizes GAN
    training and is common in progressive growing architectures.

    ``x[i] = x[i] / sqrt(mean(x[i]^2))``

    Example::

        norm = PixelNorm()
        # input:  (N, C, H, W) -> output: (N, C, H, W)
        # each channel value divided by sqrt(sum_of_squares_across_channels)
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.size(1)
        mean_sq = (x ** 2).sum(dim=1, keepdim=True)
        return x / (self.eps + mean_sq).sqrt()


class AdaINLayer(nn.Module):
    """Adaptive Instance Normalization.

    Normalizes the content tensor by its own mean/std, then reparameterizes
    using the mean/std of a style tensor.

    ``z = (x - mu_x) / sigma_x``
    ``out = z * sigma_s + mu_s``

    Example::

        adain = AdaINLayer()
        content = torch.randn(32, 64, 28, 28)   # (N, C, H, W)
        style   = torch.randn(32, 64, 28, 28)   # (N, C, H, W)
        out = adain((content, style))             # (N, 64, 28, 28)
    """

    def forward(
        self,
        inputs: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        x, s = inputs  # (N, C, H, W), (N, C, H, W)

        mu_x = x.mean(dim=[2, 3], keepdim=True)
        sigma_x = x.std(dim=[2, 3], keepdim=True)
        z = (x - mu_x) / sigma_x

        mu_s = s.mean(dim=[2, 3], keepdim=True)
        sigma_s = s.std(dim=[2, 3], keepdim=True)

        return z * sigma_s + mu_s


class MeanStdNorm(nn.Module):
    """Learnable mean/std normalization per channel.

    Similar to image-net standardization but with **learnable** parameters
    per channel.  Useful when the normalization statistics should adapt
    during training rather than being fixed.

    ``out = (x - mu) / (sigma + eps)``

    Example::

        norm = MeanStdNorm(channels=64)
        # mu and std are nn.Parameter of shape (64, 1, 1)
        out = norm(x)
    """

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(channels, 1, 1))
        self.std = nn.Parameter(torch.ones(channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mu) / (self.std + self.eps)


class NormalizationStack2d(nn.Module):
    """Compose multiple normalization layers into one module.

    Chains BatchNorm and PixelNorm (or any other nn.Module) in sequence.

    Example::

        stack = NormalizationStack2d(
            norm_types=["batch", "pixel"],
            num_features=64
        )
        # Equivalent to: nn.BatchNorm2d(64) -> PixelNorm()
    """

    def __init__(
        self,
        norm_types: list[str] = None,
        num_features: int | None = None,
        affine: bool = True,
    ) -> None:
        super().__init__()
        if norm_types is None:
            norm_types = []

        layers = []
        for norm_type in norm_types:
            if norm_type == "batch" or norm_type == "batchnorm":
                if num_features is not None:
                    layers.append(nn.BatchNorm2d(num_features, affine=affine))
                else:
                    layers.append(nn.LazyBatchNorm2d(affine=affine))
            elif norm_type == "pixel":
                layers.append(PixelNorm())
            elif norm_type == "instancenorm" or norm_type == "instance":
                layers.append(nn.InstanceNorm2d(num_features if num_features else 0))
            else:
                raise ValueError(f"Unsupported norm type: {norm_type}")

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
