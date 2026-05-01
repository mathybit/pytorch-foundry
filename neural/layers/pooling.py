"""
Pooling Layers
==============
PyTorch pooling layers for sequence-to-vector conversion with masking support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedAveragePooling(nn.Module):
    """Compute average pooling over sequence dimension, ignoring masked positions.

    Useful for converting variable-length sequences to fixed-size vectors
    when padding is used.

    Args:
        dim: Dimension to pool over (default: 1, assumes shape [B, L, D])
        eps: Small constant for numerical stability

    Input:
        x: Tensor of shape (batch, seq_len, embed_dim)
        mask: Boolean tensor of shape (batch, seq_len), True = valid, False = padding

    Output:
        Tensor of shape (batch, embed_dim)

    Example::

        pooling = MaskedAveragePooling()
        x       = torch.randn(32, 20, 128)
        mask    = torch.ones(32, 20, dtype=torch.bool)
        mask[:, 15:] = False
        out     = pooling(x, mask)  # (32, 128)
    """

    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply masked mean pooling.

        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            mask: Boolean mask (batch, seq_len), True = valid position

        Returns:
            Pooled tensor (batch, embed_dim)
        """
        mask = mask.unsqueeze(-1).float()
        x = x * mask

        summed = x.sum(dim=self.dim)
        counts = mask.sum(dim=self.dim)
        counts = torch.clamp(counts, min=self.eps)
        mean = summed / counts

        return mean


class AttentionPooling(nn.Module):
    """Attention-based pooling with a learnable query vector.

    Uses a single learnable query to compute attention weights over the sequence,
    then returns weighted sum. Handles masking to ignore padding positions.

    Args:
        embed_dim: Dimension of input embeddings
        num_heads: Number of attention heads (default: 1)
        dropout: Dropout probability for attention weights (default: 0.0)

    Input:
        x: Tensor of shape (batch, seq_len, embed_dim)
        mask: Boolean tensor of shape (batch, seq_len), True = valid, False = padding

    Output:
        Tensor of shape (batch, embed_dim)

    Example::

        pooling = AttentionPooling(embed_dim=128, num_heads=4)
        x       = torch.randn(32, 20, 128)
        mask    = torch.ones(32, 20, dtype=torch.bool)
        mask[:, 15:] = False
        out     = pooling(x, mask)  # (32, 128)
    """

    def __init__(self, embed_dim: int, num_heads: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize parameters using Xavier uniform."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply attention-based pooling.

        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            mask: Boolean mask (batch, seq_len), True = valid position

        Returns:
            Pooled tensor (batch, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape

        query = self.query.expand(batch_size, -1, -1)

        Q = self.q_proj(query)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)

        context = context.transpose(1, 2).contiguous().view(batch_size, 1, embed_dim)
        output = self.out_proj(context)
        output = output.squeeze(1)

        return output


class MaskedMaxPooling(nn.Module):
    """Max pooling over sequence dimension, ignoring masked positions.

    Args:
        dim: Dimension to pool over (default: 1)

    Input:
        x: Tensor of shape (batch, seq_len, embed_dim)
        mask: Boolean tensor of shape (batch, seq_len), True = valid

    Output:
        Tensor of shape (batch, embed_dim)

    Example::

        pooling = MaskedMaxPooling()
        x       = torch.randn(32, 20, 128)
        mask    = torch.ones(32, 20, dtype=torch.bool)
        out     = pooling(x, mask)  # (32, 128)
    """

    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply masked max pooling."""
        mask = mask.unsqueeze(-1).float()
        masked_x = x.masked_fill(~mask.bool(), float("-inf"))
        pooled, _ = masked_x.max(dim=self.dim)
        return pooled


class CrossChannelPooling2d(nn.Module):
    """Pool over the channel dimension at each spatial position (2-D only).

    Standard pooling layers pool over spatial dimensions (H, W), reducing
    the output to (N, C).  This layer does the opposite: at each spatial
    position it computes per-channel statistics (mean and std) across the
    channel dimension, producing (N, 2, 1, 1) output.

    Parameters:
        channels:  number of input channels (required when learnable=True).
        learnable: if True, uses softmax-normalized learned weights instead
                   of uniform averaging across channels.
        eps:       numerical stability term for std computation.

    Example (2x2 grid, 3 channels)::

        # Input shape: (1, 3, 2, 2)
        # Channel 0: [[1, 2], [3, 4]]
        # Channel 1: [[5, 6], [7, 8]]
        # Channel 2: [[9, 10], [11, 12]]
        #
        # At spatial position (0,0): values are [1, 5, 9]
        #   mean = 5.0, std ~ 4.0
        # At spatial position (0,1): values are [2, 6, 10]
        #   mean = 6.0, std ~ 4.0
        # ...etc

    Output: [mean_channels, std_channels] concatenated -> (N, 2, 1, 1)
    """

    def __init__(
        self,
        channels: int | None = None,
        learnable: bool = False,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.learnable = learnable
        self.eps = eps
        if learnable:
            if channels is None:
                raise ValueError("channels is required when learnable=True")
            self.weight = nn.Parameter(torch.zeros(channels))
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.learnable:
            w = F.softmax(self.weight, dim=0)       # (C,)
            w = w.unsqueeze(1).unsqueeze(1)          # (C, 1, 1)
            mu = (x * w).sum(dim=1, keepdim=True)    # (N, 1, 1, 1)
            var = ((x - mu) ** 2 * w).sum(dim=1, keepdim=True)
            sigma = var.sqrt()
            return torch.cat([mu, sigma], dim=1)     # (N, 2, 1, 1)

        mu = x.mean(dim=1, keepdim=True)    # (N, 1, 1, 1)
        sigma = x.std(dim=1, keepdim=True)  # (N, 1, 1, 1)
        return torch.cat([mu, sigma], dim=1)  # (N, 2, 1, 1)


CrossChannelPool2d = CrossChannelPooling2d
