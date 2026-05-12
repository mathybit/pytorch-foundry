"""Tests for all neural.layers classes."""

import math
from pathlib import Path
import torch
import torch.nn as nn
import pytest
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from neural.layers.convolutional import (
    MultiDilationConv1d,
    MultiDilationConv2d,
    MultiDilationConv3d,
    SubPixelConv2d,
)
from neural.layers.normalization import (
    PixelNorm,
    AdaINLayer,
    MeanStdNorm,
    NormalizationStack2d,
)
from neural.layers.residual import (
    SimpleResidualBlock,
    ResidualBlock,
    BottleneckResidualBlock,
    ResidualStack,
)
from neural.layers.pooling import (
    AttentionPooling,
    MaskedAveragePooling,
    MaskedMaxPooling,
    CrossChannelPooling2d,
    CrossChannelPool2d,
)
from neural.layers.activations import (
    Scaling,
    Mish,
    Swish,
    GELU,
    GLU,
    PReLU,
    SiLU,
    Activation,
)
from neural.layers.temporal import (
    PositionalEncoding,
    LearnedPositionalEncoding,
    Time2Vec,
    CausalConv1d,
    WaveNet,
    RotaryPositionalEncoding,
)
from neural.layers.backbones import PretrainedBackbone
from neural.layers.output_heads import ClassificationHead, RegressionHead


# ---------- Convolutional Layers ----------


class TestMultiDilationConv1d:
    def test_single_dilation(self):
        block = MultiDilationConv1d(16, 32, kernel_size=3, dilations=[1])
        x = torch.randn(2, 16, 64)
        out = block(x)
        assert out.shape == (2, 32, 64)

    def test_multiple_dilations(self):
        block = MultiDilationConv1d(16, 48, kernel_size=3, dilations=[1, 2, 4])
        x = torch.randn(2, 16, 64)
        out = block(x)
        assert out.shape == (2, 48, 64)

    def test_list_kernel_size(self):
        block = MultiDilationConv1d(16, 32, kernel_size=[3, 5], dilations=[1, 2])
        x = torch.randn(2, 16, 64)
        out = block(x)
        assert out.shape == (2, 32, 64)

    def test_out_channels_not_divisible(self):
        with pytest.raises(ValueError, match="out_channels"):
            MultiDilationConv1d(16, 17, kernel_size=3, dilations=[1, 2])

    def test_invalid_padding(self):
        with pytest.raises(ValueError, match="padding"):
            MultiDilationConv1d(16, 32, padding="valid")


class TestMultiDilationConv2d:
    def test_single_dilation(self):
        block = MultiDilationConv2d(32, 64, kernel_size=3, dilations=[1])
        x = torch.randn(2, 32, 28, 28)
        out = block(x)
        assert out.shape == (2, 64, 28, 28)

    def test_multiple_dilations(self):
        block = MultiDilationConv2d(32, 64, kernel_size=3, dilations=[1, 2])
        x = torch.randn(2, 32, 28, 28)
        out = block(x)
        assert out.shape == (2, 64, 28, 28)


class TestMultiDilationConv3d:
    def test_single_dilation(self):
        block = MultiDilationConv3d(8, 16, kernel_size=3, dilations=[1])
        x = torch.randn(2, 8, 4, 28, 28)
        out = block(x)
        assert out.shape == (2, 16, 4, 28, 28)

    def test_multiple_dilations(self):
        block = MultiDilationConv3d(8, 16, kernel_size=3, dilations=[1, 2])
        x = torch.randn(2, 8, 4, 28, 28)
        out = block(x)
        assert out.shape == (2, 16, 4, 28, 28)


class TestSubPixelConv2d:
    def test_2x_upscale(self):
        up = SubPixelConv2d(64, 32, scale_factor=2)
        x = torch.randn(4, 64, 14, 14)
        out = up(x)
        assert out.shape == (4, 32, 28, 28)

    def test_multi_branch(self):
        up = SubPixelConv2d(64, 32, scale_factor=2, kernel_size=[3, 5], dilations=[1, 2])
        x = torch.randn(4, 64, 14, 14)
        out = up(x)
        assert out.shape == (4, 32, 28, 28)

    def test_out_channels_not_divisible(self):
        with pytest.raises(ValueError, match="out_channels"):
            SubPixelConv2d(64, 33, scale_factor=2, dilations=[1, 2])


# ---------- Normalization Layers ----------


class TestPixelNorm:
    def test_output_shape(self):
        norm = PixelNorm()
        x = torch.randn(4, 64, 28, 28)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalized(self):
        norm = PixelNorm()
        x = torch.randn(4, 64, 28, 28)
        out = norm(x)
        # Each pixel should have L2 norm ~ 1
        l2 = out.norm(dim=1)  # (N, H, W)
        assert l2.mean() == pytest.approx(1.0, abs=1e-5)


class TestAdaINLayer:
    def test_output_shape(self):
        adain = AdaINLayer()
        content = torch.randn(32, 64, 28, 28)
        style = torch.randn(32, 64, 28, 28)
        out = adain((content, style))
        assert out.shape == content.shape

    def test_mean_std_applied(self):
        adain = AdaINLayer()
        content = torch.randn(32, 64, 28, 28)
        style = torch.randn(32, 64, 28, 28)
        out = adain((content, style))
        out_mean = out.mean(dim=[2, 3])
        out_std = out.std(dim=[2, 3])
        # Output should have style mean/std
        assert torch.allclose(out_mean, style.mean(dim=[2, 3]), atol=1e-5)
        assert torch.allclose(out_std, style.std(dim=[2, 3]), atol=1e-5)


class TestMeanStdNorm:
    def test_output_shape(self):
        norm = MeanStdNorm(channels=64)
        x = torch.randn(4, 64, 28, 28)
        out = norm(x)
        assert out.shape == x.shape

    def test_learnable_params(self):
        norm = MeanStdNorm(channels=64)
        assert norm.mu.shape == (64, 1, 1)
        assert norm.std.shape == (64, 1, 1)
        assert norm.mu.requires_grad


class TestNormalizationStack2d:
    def test_batch_pixel(self):
        stack = NormalizationStack2d(norm_types=["batch", "pixel"], num_features=64)
        x = torch.randn(4, 64, 28, 28)
        out = stack(x)
        assert out.shape == x.shape

    def test_instance_only(self):
        stack = NormalizationStack2d(norm_types=["instance"], num_features=64)
        x = torch.randn(4, 64, 28, 28)
        out = stack(x)
        assert out.shape == x.shape

    def test_empty(self):
        stack = NormalizationStack2d(norm_types=[])
        x = torch.randn(4, 64, 28, 28)
        out = stack(x)
        assert torch.allclose(out, x)

    def test_unsupported_type(self):
        with pytest.raises(ValueError, match="Unsupported"):
            NormalizationStack2d(norm_types=["unknown"])

    def test_lazy_batch(self):
        stack = NormalizationStack2d(norm_types=["batch"], num_features=None)
        x = torch.randn(4, 64, 28, 28)
        out = stack(x)
        assert out.shape == x.shape


# ---------- Residual Layers ----------


class TestSimpleResidualBlock:
    def test_output_shape(self):
        block = SimpleResidualBlock(64, kernel_size=3)
        x = torch.randn(4, 64, 28, 28)
        out = block(x)
        assert out.shape == x.shape

    def test_no_activation(self):
        block = SimpleResidualBlock(64)
        x = torch.ones(4, 64, 4, 4)
        out = block(x)
        # Identity + conv -> output should differ from input
        assert not torch.allclose(out, x)

    def test_with_activation(self):
        block = SimpleResidualBlock(64, activation="relu")
        x = torch.randn(4, 64, 4, 4)
        out = block(x)
        assert out.shape == x.shape


class TestResidualBlock:
    def test_output_shape(self):
        block = ResidualBlock(64, kernel_size=3)
        x = torch.randn(4, 64, 28, 28)
        out = block(x)
        assert out.shape == x.shape

    def test_params(self):
        block = ResidualBlock(channels=64, norm="batch", activation="relu")
        assert hasattr(block, "conv1")
        assert hasattr(block, "conv2")
        assert hasattr(block, "norm1")
        assert hasattr(block, "norm2")
        assert hasattr(block, "activation")


class TestBottleneckResidualBlock:
    def test_output_shape(self):
        block = BottleneckResidualBlock(channels=64, reduction=2)
        x = torch.randn(4, 64, 28, 28)
        out = block(x)
        assert out.shape == x.shape

    def test_bottleneck_channels(self):
        block = BottleneckResidualBlock(channels=64, reduction=2)
        x = torch.randn(4, 64, 28, 28)
        out = block(x)
        assert out.shape == x.shape


class TestResidualStack:
    def test_output_shape(self):
        stack = ResidualStack(64, n_blocks=3)
        x = torch.randn(4, 64, 28, 28)
        out = stack(x)
        assert out.shape == x.shape

    def test_input_projection(self):
        stack = ResidualStack(128, n_blocks=2, in_channels=64)
        x = torch.randn(4, 64, 28, 28)
        out = stack(x)
        assert out.shape == (4, 128, 28, 28)

    def test_different_block_types(self):
        for btype in ["srgan", "preact", "bottleneck"]:
            stack = ResidualStack(64, n_blocks=2, block_type=btype)
            x = torch.randn(4, 64, 4, 4)
            out = stack(x)
            assert out.shape == x.shape


# ---------- Pooling Layers ----------


class TestAttentionPooling:
    def test_output_shape(self):
        pool = AttentionPooling(embed_dim=128)
        x = torch.randn(32, 20, 128)
        mask = torch.ones(32, 20, dtype=torch.bool)
        out = pool(x, mask)
        assert out.shape == (32, 128)

    def test_with_padding(self):
        pool = AttentionPooling(embed_dim=64, num_heads=4)
        x = torch.randn(8, 10, 64)
        mask = torch.ones(8, 10, dtype=torch.bool)
        mask[:, 7:] = False
        out = pool(x, mask)
        assert out.shape == (8, 64)

    def test_no_mask(self):
        pool = AttentionPooling(embed_dim=64)
        x = torch.randn(8, 10, 64)
        out = pool(x, None)
        assert out.shape == (8, 64)


class TestMaskedAveragePooling:
    def test_output_shape(self):
        pool = MaskedAveragePooling()
        x = torch.randn(32, 20, 128)
        mask = torch.ones(32, 20, dtype=torch.bool)
        out = pool(x, mask)
        assert out.shape == (32, 128)

    def test_ignores_mask(self):
        pool = MaskedAveragePooling()
        x = torch.ones(4, 10, 8)
        mask = torch.ones(4, 10, dtype=torch.bool)
        mask[:, 7:] = False
        out = pool(x, mask)
        assert torch.allclose(out, torch.ones(4, 8))

    def test_different_dim(self):
        pool = MaskedAveragePooling(dim=2)
        x = torch.randn(4, 8, 20)
        mask = torch.ones(4, 20, dtype=torch.bool)
        out = pool(x, mask)
        assert out.shape == (4, 8)


class TestMaskedMaxPooling:
    def test_output_shape(self):
        pool = MaskedMaxPooling()
        x = torch.randn(32, 20, 128)
        mask = torch.ones(32, 20, dtype=torch.bool)
        out = pool(x, mask)
        assert out.shape == (32, 128)

    def test_ignores_mask(self):
        pool = MaskedMaxPooling()
        x = torch.zeros(4, 10, 8)
        x[:, :7, :] = 1.0
        mask = torch.ones(4, 10, dtype=torch.bool)
        mask[:, 7:] = False
        out = pool(x, mask)
        assert torch.allclose(out, torch.ones(4, 8))

    def test_all_masked(self):
        pool = MaskedMaxPooling()
        x = torch.zeros(4, 10, 8)
        mask = torch.zeros(4, 10, dtype=torch.bool)
        out = pool(x, mask)
        assert torch.all(torch.isinf(out) & (out < 0))


class TestCrossChannelPooling2d:
    def test_output_shape_basic(self):
        pool = CrossChannelPooling2d()
        x = torch.randn(4, 64, 28, 28)
        out = pool(x)
        # Non-trainable: mean/std across channels at each spatial position
        assert out.shape == (4, 2, 28, 28)

    def test_trainable(self):
        pool = CrossChannelPooling2d(channels=64, trainable=True)
        x = torch.randn(4, 64, 28, 28)
        out = pool(x)
        assert out.shape == (4, 2, 28, 28)
        assert hasattr(pool, "weight")
        assert pool.weight.requires_grad

    def test_alias(self):
        pool = CrossChannelPool2d()
        x = torch.randn(4, 64, 28, 28)
        out = pool(x)
        assert out.shape == (4, 2, 28, 28)

    def test_trainable_requires_channels(self):
        with pytest.raises(ValueError, match="channels is required"):
            CrossChannelPooling2d(trainable=True)


# ---------- Activation Layers ----------


class TestScaling:
    def test_output_shape(self):
        s = Scaling(channels=64)
        x = torch.randn(4, 64, 28, 28)
        out = s(x)
        assert out.shape == x.shape

    def test_init_value(self):
        s = Scaling(channels=64, init=1.0)
        x = torch.randn(4, 64, 28, 28)
        out = s(x)
        assert torch.allclose(out, x)

    def test_scalar(self):
        s = Scaling(channels=1)
        x = torch.randn(4, 64, 28, 28)
        out = s(x)
        assert out.shape == x.shape


class TestMish:
    def test_output_shape(self):
        act = Mish()
        x = torch.randn(4, 64, 28, 28)
        out = act(x)
        assert out.shape == x.shape

    def test_identity_near_zero(self):
        act = Mish()
        x = torch.zeros(4, 64, 28, 28)
        out = act(x)
        assert torch.allclose(out, torch.zeros_like(x))

    def test_monotonicity(self):
        act = Mish()
        x = torch.linspace(-5, 5, 1000).unsqueeze(-1)
        out = act(x)
        assert torch.all(out[500:] >= out[:500])


class TestSwish:
    def test_output_shape(self):
        act = Swish()
        x = torch.randn(4, 64, 28, 28)
        out = act(x)
        assert out.shape == x.shape

    def test_trainable_beta(self):
        act = Swish(trainable=True)
        assert act.beta.requires_grad

    def test_fixed_beta(self):
        act = Swish(trainable=False, beta=1.0)
        assert not act.beta.requires_grad


class TestGELU:
    def test_output_shape(self):
        act = GELU()
        x = torch.randn(4, 64, 28, 28)
        out = act(x)
        assert out.shape == x.shape

    def test_monotonicity(self):
        act = GELU()
        x = torch.linspace(-5, 5, 1000).unsqueeze(-1)
        out = act(x)
        assert torch.all(out[500:] >= out[:500])


class TestGLU:
    def test_output_shape(self):
        glu = GLU(dim=1)
        x = torch.randn(4, 64, 28, 28)
        out = glu(x)
        assert out.shape == (4, 32, 28, 28)

    def test_different_dim(self):
        glu = GLU(dim=0)
        x = torch.randn(64, 4, 28, 28)
        out = glu(x)
        assert out.shape == (32, 4, 28, 28)


class TestPReLU:
    def test_output_shape(self):
        act = PReLU()
        x = torch.randn(4, 64, 28, 28)
        out = act(x)
        assert out.shape == x.shape

    def test_learnable_alpha(self):
        act = PReLU(num_params=64)
        assert act.weight.requires_grad

    def test_shared_alpha(self):
        act = PReLU(num_params=1)
        x = torch.randn(4, 64, 4, 4)
        out = act(x)
        assert out.shape == x.shape


class TestSiLU:
    def test_output_shape(self):
        act = SiLU()
        x = torch.randn(4, 64, 28, 28)
        out = act(x)
        assert out.shape == x.shape

    def test_same_as_swish_1(self):
        act_silu = SiLU()
        act_swish = Swish(trainable=False, beta=1.0)
        x = torch.randn(4, 64, 28, 28)
        assert torch.allclose(act_silu(x), act_swish(x))


class TestActivation:
    def test_relu(self):
        act = Activation("relu")
        x = torch.randn(4, 64, 28, 28)
        out = act(x)
        assert out.shape == x.shape

    def test_case_insensitive(self):
        act1 = Activation("relu")
        act2 = Activation("ReLU")
        x = torch.randn(4, 64, 4, 4)
        out1 = act1(x)
        out2 = act2(x)
        assert torch.allclose(out1, out2)

    def test_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            Activation("unknown")

    def test_wrong_kwargs(self):
        with pytest.raises(TypeError, match="unexpected"):
            Activation("relu", alpha=0.1)

    def test_swish(self):
        act = Activation("swish", trainable=False, beta=1.0)
        x = torch.randn(4, 64, 4, 4)
        out = act(x)
        assert out.shape == x.shape

    def test_silu(self):
        act = Activation("silu")
        x = torch.randn(4, 64, 4, 4)
        out = act(x)
        assert out.shape == x.shape

    def test_glu(self):
        act = Activation("glu", dim=1)
        x = torch.randn(4, 64, 28, 28)
        out = act(x)
        assert out.shape == (4, 32, 28, 28)

    def test_prelu(self):
        act = Activation("prelu", num_params=64)
        x = torch.randn(4, 64, 4, 4)
        out = act(x)
        assert out.shape == x.shape

    def test_leakyrelu(self):
        act = Activation("leakyrelu", alpha=0.01)
        x = torch.randn(4, 64, 4, 4)
        out = act(x)
        assert out.shape == x.shape

    def test_tanh(self):
        act = Activation("tanh")
        x = torch.randn(4, 64, 4, 4)
        out = act(x)
        assert out.shape == x.shape
        assert out.max() <= 1.0

    def test_sigmoid(self):
        act = Activation("sigmoid")
        x = torch.randn(4, 64, 4, 4)
        out = act(x)
        assert out.shape == x.shape
        assert out.max() <= 1.0
        assert out.min() >= 0.0

    def test_elu(self):
        act = Activation("elu", alpha=1.0)
        x = torch.randn(4, 64, 4, 4)
        out = act(x)
        assert out.shape == x.shape

    def test_celu(self):
        act = Activation("celu", alpha=1.0)
        x = torch.randn(4, 64, 4, 4)
        out = act(x)
        assert out.shape == x.shape

    def test_mish(self):
        act = Activation("mish")
        x = torch.randn(4, 64, 4, 4)
        out = act(x)
        assert out.shape == x.shape

    def test_hardtanh(self):
        act = Activation("hardtanh", min_val=-1.0, max_val=1.0)
        x = torch.randn(4, 64, 4, 4) * 100
        out = act(x)
        assert out.max() <= 1.0
        assert out.min() >= -1.0

    def test_hardswish(self):
        act = Activation("hardswish")
        x = torch.randn(4, 64, 4, 4)
        out = act(x)
        assert out.shape == x.shape


# ---------- Temporal Layers ----------


class TestPositionalEncoding:
    def test_output_shape(self):
        pe = PositionalEncoding(128, max_len=5000)
        x = torch.randn(4, 20, 128)
        out = pe(x)
        assert out.shape == x.shape

    def test_pe_has_values(self):
        pe = PositionalEncoding(128, max_len=20)
        x = torch.zeros(4, 20, 128)
        out = pe(x)
        assert not torch.allclose(out, torch.zeros_like(out))


class TestLearnedPositionalEncoding:
    def test_output_shape(self):
        pe = LearnedPositionalEncoding(128, max_len=5000)
        x = torch.randn(4, 20, 128)
        out = pe(x)
        assert out.shape == x.shape

    def test_learnable(self):
        pe = LearnedPositionalEncoding(128, max_len=20)
        assert pe.pe.requires_grad


class TestTime2Vec:
    def test_output_shape(self):
        t2v = Time2Vec(input_dim=1, output_dim=64)
        t = torch.randn(4, 20, 1)
        out = t2v(t)
        assert out.shape == (4, 20, 64)

    def test_linear_first_component(self):
        t2v = Time2Vec(input_dim=1, output_dim=64)
        t = torch.tensor([[[0.0]], [[1.0]], [[2.0]]])
        out = t2v(t)
        # First component should be linear: w*t + phi
        first = out[..., 0]
        diffs = first[1:] - first[:-1]
        assert torch.allclose(diffs[0], diffs[1])  # linear → constant diffs


class TestCausalConv1d:
    def test_output_shape(self):
        conv = CausalConv1d(64, 128, kernel_size=3)
        x = torch.randn(4, 64, 32)
        out = conv(x)
        assert out.shape == (4, 128, 32)

    def test_causal(self):
        conv = CausalConv1d(64, 128, kernel_size=3)
        x = torch.randn(4, 64, 32)
        out = conv(x)
        # Output should be causal (not future-dependent)
        # Test: zero out past, check output changes
        x_padded = torch.zeros_like(x)
        x_padded[:, :, -16:] = x[:, :, -16:]
        out_full = conv(x)
        out_padded = conv(x_padded)
        # Output[j] depends on input[j-pad], input[j-pad+1], input[j-pad+2]
        # With pad=2, need j-2 >= 16 → j >= 18 for full receptive field in non-zero region
        assert torch.equal(out_full[:, :, 18:], out_padded[:, :, 18:])

    def test_dilation(self):
        conv = CausalConv1d(64, 128, kernel_size=3, dilation=2)
        x = torch.randn(4, 64, 32)
        out = conv(x)
        assert out.shape == (4, 128, 32)


class TestWaveNet:
    def test_output_shape(self):
        w = WaveNet(128, filters=64, kernel_size=3, dilations=(1, 2, 4))
        x = torch.randn(4, 128, 32)
        residual, skip = w(x)
        assert residual.shape == x.shape
        assert skip.shape == (4, 192, 32)  # 3 * 64

    def test_skip_summation(self):
        w = WaveNet(128, filters=32, kernel_size=3, dilations=(1, 2))
        x = torch.randn(4, 128, 32)
        residual, skip = w(x)
        assert residual.shape == x.shape
        assert skip.shape == (4, 64, 32)  # 2 * 32


class TestRotaryPositionalEncoding:
    def test_output_shape(self):
        rope = RotaryPositionalEncoding(64, base=10000.0)
        q = torch.randn(2, 4, 10, 64)
        k = torch.randn(2, 4, 10, 64)
        q_out, k_out = rope(q, k)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_rotated(self):
        rope = RotaryPositionalEncoding(64, base=10000.0)
        q = torch.randn(2, 4, 10, 64)
        k = torch.randn(2, 4, 10, 64)
        q_out, k_out = rope(q, k)
        # Rotation preserves norm
        assert torch.allclose(q.norm(dim=-1), q_out.norm(dim=-1), atol=1e-5)
        assert torch.allclose(k.norm(dim=-1), k_out.norm(dim=-1), atol=1e-5)

    def test_head_dim_too_small(self):
        with pytest.raises(ValueError, match="head_dim must be >= 2"):
            RotaryPositionalEncoding(1)


# ---------- Backbone Layers ----------


class TestPretrainedBackbone:
    def test_freeze_unfreeze(self):
        backbone = PretrainedBackbone("resnet18", pretrained=False)
        assert not backbone.frozen
        backbone.freeze()
        assert backbone.frozen
        for p in backbone.model.parameters():
            assert not p.requires_grad
        backbone.unfreeze()
        assert not backbone.frozen
        for p in backbone.model.parameters():
            assert p.requires_grad

    def test_freeze_pretrained_layers(self):
        backbone = PretrainedBackbone("resnet18", pretrained=False)
        backbone.freeze_pretrained_layers()
        # BN layers should still be trainable
        bn_count = 0
        trainable_bn = 0
        for name, module in backbone.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                bn_count += 1
                for p in module.parameters():
                    if p.requires_grad:
                        trainable_bn += 1
        assert bn_count > 0
        assert trainable_bn > 0

    def test_unknown_model(self):
        with pytest.raises(ValueError, match="Unknown model"):
            PretrainedBackbone("unknown_model")

    def test_unknown_output_layer(self):
        backbone = PretrainedBackbone("resnet18", pretrained=False)
        with pytest.raises(ValueError, match="Unknown output_layer"):
            backbone.output_layer = "unknown"

    def test_in_channels_override(self):
        backbone = PretrainedBackbone("resnet18", pretrained=False, in_channels=1)
        x = torch.randn(2, 1, 224, 224)
        out = backbone(x)
        assert out.shape == (2, backbone.num_channels)

    def test_forward_resnet(self):
        backbone = PretrainedBackbone("resnet18", pretrained=False, in_channels=3)
        x = torch.randn(2, 3, 224, 224)
        out = backbone(x)
        assert out.shape == (2, backbone.num_channels)

    def test_forward_efficientnet(self):
        backbone = PretrainedBackbone("efficientnet_b0", pretrained=False, in_channels=3)
        x = torch.randn(2, 3, 224, 224)
        out = backbone(x)
        assert out.shape == (2, backbone.num_channels)


# ---------- Output Heads ----------


class TestClassificationHead:
    def test_output_shape(self):
        head = ClassificationHead(512, 256, n_classes=10)
        x = torch.randn(4, 512)
        out = head(x)
        assert out.shape == (4, 10)

    def test_embedding(self):
        head = ClassificationHead(512, 256, n_classes=10)
        x = torch.randn(4, 512)
        emb = head(x, embedding=True)
        assert emb.shape == (4, 256)

    def test_binary(self):
        head = ClassificationHead(512, 256, n_classes=2)
        x = torch.randn(4, 512)
        out = head(x)
        assert out.shape == (4, 2)

    def test_unknown_activation(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            ClassificationHead(512, 256, n_classes=10, activation="unknown")

    def test_different_activations(self):
        for act in ["relu", "elu", "gelu", "sigmoid", "tanh"]:
            head = ClassificationHead(64, 32, n_classes=5, activation=act)
            x = torch.randn(4, 64)
            out = head(x)
            assert out.shape == (4, 5)


class TestRegressionHead:
    def test_output_shape(self):
        head = RegressionHead(512, 256)
        x = torch.randn(4, 512)
        out = head(x)
        assert out.shape == (4, 1)

    def test_embedding(self):
        head = RegressionHead(512, 256)
        x = torch.randn(4, 512)
        emb = head(x, embedding=True)
        assert emb.shape == (4, 256)

    def test_different_activations(self):
        for act in ["relu", "elu", "sigmoid"]:
            head = RegressionHead(64, 32, activation=act)
            x = torch.randn(4, 64)
            out = head(x)
            assert out.shape == (4, 1)

    def test_unknown_activation(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            RegressionHead(512, 256, activation="unknown")
