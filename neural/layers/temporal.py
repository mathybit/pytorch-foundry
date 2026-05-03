"""
Temporal layers for sequence and time-series modeling.

Provides PositionalEncoding, Time2Vec, WaveNet, RotaryPositionalEncoding,
and LearnedPositionalEncoding.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding.

    Adds sin/cos frequencies at different scales to the input embeddings.
    Original from "Attention is All You Need."

    Example::

        pe = PositionalEncoding(128, max_len=5000)
        # x: (batch, seq_len, 128)
        out = pe(x)  # (batch, seq_len, 128)
    """

    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float()
            * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding. x: (batch, seq_len, embed_dim)."""
        return x + self.pe[:, :x.size(1)]


class LearnedPositionalEncoding(nn.Module):
    """Learned position embeddings.

    Learns a separate embedding vector for each position. Added to input.

    Example::

        pe = LearnedPositionalEncoding(128, max_len=5000)
        out = pe(x)
    """

    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add learned positional encoding. x: (batch, seq_len, embed_dim)."""
        return x + self.pe[:, :x.size(1)]


class Time2Vec(nn.Module):
    """Time2Vec activation for temporal feature transformation.

    Applies a mix of linear and sinusoidal transformations to temporal inputs:
    - First component: w * t + phi (linear)
    - Remaining components: sin(s * t + phi) (sinusoidal)

    Useful for embedding continuous time features in time-series models.

    Example::

        t2v = Time2Vec(input_dim=1, output_dim=64)
        # t: (batch, seq_len, 1) -> out: (batch, seq_len, 64)
        out = t2v(t)
    """

    def __init__(self, input_dim: int = 1, output_dim: int = 64) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # First output component is linear -> weight + bias
        self.lin_w = nn.Parameter(torch.randn(input_dim, 1) * 0.02)
        self.lin_b = nn.Parameter(torch.zeros(1))

        # Remaining components are sinusoidal -> weight + bias
        n_sinusoidal = output_dim - 1
        self.sin_w = nn.Parameter(torch.randn(n_sinusoidal, input_dim) * 0.02)
        self.sin_b = nn.Parameter(torch.zeros(n_sinusoidal))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Transform temporal features. t: (..., input_dim)."""
        lin = (t @ self.lin_w).squeeze(-1).unsqueeze(-1)
        sin = torch.sin(t @ self.sin_w.T + self.sin_b)
        return torch.cat([lin, sin], dim=-1)


class CausalConv1d(nn.Module):
    """1-D causal (left-padded) convolution.

    Pads only the left (past) side so that output at time t depends only
    on inputs up to time t.  Useful for autoregressive models.

    Example::

        conv = CausalConv1d(64, 128, kernel_size=3)
        # x: (N, 64, T) -> out: (N, 128, T)
        out = conv(x)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.pad = nn.ConstantPad1d((pad, 0), 0.0)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, C, T) -> (N, out_channels, T)."""
        return self.conv(self.pad(x))


class WaveNet(nn.Module):
    """WaveNet residual block with causal dilated convolution.

    Contains a gated activation unit (GAU) with causal dilated convolutions:
    - Tanh gate and Sigmoid gate applied in parallel
    - Gate outputs multiplied element-wise
    - Residual connection adds input to output
    - Skip connection feeds forward (concatenated across dilations)

    Example::

        w = WaveNet(128, filters=64, kernel_size=3, dilations=[1, 2, 4])
        # x: (N, 128, T) -> residual: (N, 128, T), skip: (N, n*dilated*filters, T)
        residual, skip = w(x)
    """

    def __init__(
        self,
        channels: int,
        filters: int,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4),
    ) -> None:
        super().__init__()
        self.channels = channels
        self.filters = filters
        n = len(dilations)

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        for d in dilations:
            fc = CausalConv1d(channels, filters, kernel_size, dilation=d)
            gc = CausalConv1d(channels, filters, kernel_size, dilation=d)
            sc = nn.Conv1d(filters, filters, 1)
            self.filter_convs.append(fc)
            self.gate_convs.append(gc)
            self.skip_convs.append(sc)

        self.res_conv = nn.Conv1d(n * filters, channels, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (N, C, T) -> (residual, skip) each (N, C, T), (N, n*filters, T)."""
        skip = []
        for fc, gc, sc in zip(self.filter_convs, self.gate_convs, self.skip_convs):
            f = torch.tanh(fc(x))
            g = torch.sigmoid(gc(x))
            z = f * g  # gated output
            skip.append(sc(z))

        skip = torch.cat(skip, dim=1)
        residual = self.res_conv(skip)
        return residual + x, skip


class RotaryPositionalEncoding(nn.Module):
    """Rotary Positional Encoding (RoPE).

    Rotates query and key vectors by learned angles at each position.
    Used in PaLM, GPT-J, and other modern transformers.

    Example::

        rope = RotaryPositionalEncoding(128, base=10000.0)
        # q: (batch, heads, seq_len, head_dim)
        # k: (batch, heads, seq_len, head_dim)
        q_rot, k_rot = rope(q, k)
    """

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base

        if head_dim < 2:
            raise ValueError("head_dim must be >= 2")

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _build_cache(self, seq_len: int) -> None:
        if seq_len > self._seq_len_cached:
            t = torch.arange(seq_len)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            self._cos_cached = freqs.cos()
            self._sin_cached = freqs.sin()
            self._seq_len_cached = seq_len

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to query and key tensors.

        Args:
            q: (batch, heads, seq_len, head_dim)
            k: (batch, heads, seq_len, head_dim)

        Returns:
            Rotated q and k tensors.
        """
        self._build_cache(q.size(2))

        cos = self._cos_cached[:q.size(2)].unsqueeze(0).unsqueeze(0)  # (1, 1, L, D/2)
        sin = self._sin_cached[:q.size(2)].unsqueeze(0).unsqueeze(0)

        # Split into even/odd pairs
        q_even, q_odd = q[..., 0::2], q[..., 1::2]
        k_even, k_odd = k[..., 0::2], k[..., 1::2]

        # Rotate: (x, y) -> (x*cos - y*sin, x*sin + y*cos)
        q_rot = torch.cat([q_even * cos - q_odd * sin, q_even * sin + q_odd * cos], dim=-1)
        k_rot = torch.cat([k_even * cos - k_odd * sin, k_even * sin + k_odd * cos], dim=-1)

        return q_rot, k_rot
