# Neural Module Documentation

## Table of Contents

- [Layers](#layers)
  - [Convolutional Layers](#convolutional-layers)
  - [Normalization Layers](#normalization-layers)
  - [Residual Layers](#residual-layers)
  - [Pooling Layers](#pooling-layers)
  - [Activation Layers](#activation-layers)
  - [Temporal Layers](#temporal-layers)
  - [Backbone Layers](#backbone-layers)
  - [Output Heads](#output-heads)
- [Datasets](#datasets)
- [Losses](#losses)
- [Evaluation](#evaluation)
- [Training](#training)
- [Utils](#utils)

---

## Layers

### Convolutional Layers

#### MultiDilationConv1d

Parallel dilated 1-D convolutions with channels split evenly across branches.

```python
from neural.layers import MultiDilationConv1d

# Dual-dilation block: receptive fields of 3 and 5
block = MultiDilationConv1d(64, 128, kernel_size=3, dilations=[1, 2])
# out_channels = 128, out_channels_per_conv = 64
out = block(x)  # (N, 64, T) -> (N, 128, T)

# Multi-branch with per-branch kernel sizes
block = MultiDilationConv1d(64, 96, kernel_size=[3, 5], dilations=[1, 2, 4])
out = block(x)
```

Parameters: `in_channels`, `out_channels`, `kernel_size` (int or list), `padding="same"`, `dilations`, `bias=True`, `stride=1`

---

#### MultiDilationConv2d

Same as MultiDilationConv1d but for image data (N, C, H, W).

```python
from neural.layers import MultiDilationConv2d

block = MultiDilationConv2d(32, 64, kernel_size=3, dilations=[1, 2, 4])
out = block(x)  # (N, 32, H, W) -> (N, 64, H, W)
```

---

#### MultiDilationConv3d

Same as MultiDilationConv1d but for volumetric data (N, C, D, H, W).

```python
from neural.layers import MultiDilationConv3d

block = MultiDilationConv3d(1, 32, kernel_size=3, dilations=[1, 2])
out = block(x)  # (N, 1, D, H, W) -> (N, 32, D, H, W)
```

---

#### SubPixelConv2d

Sub-pixel upscaling convolution using pixel shuffle. Applies parallel convolutions with different kernel sizes/dilations, then rearranges channels into spatial dimensions.

```python
from neural.layers import SubPixelConv2d

# Upscale 2x, single kernel
up = SubPixelConv2d(64, 32, scale_factor=2)
out = up(x)  # (N, 64, H, W) -> (N, 32, 2*H, 2*W)

# Multi-branch with per-branch kernel sizes
up = SubPixelConv2d(64, 32, scale_factor=2, kernel_size=[3, 5], dilations=[1, 2])
out = up(x)
```

---

### Normalization Layers

#### PixelNorm

Pixel-wise L2 normalization. Divides each pixel by the L2 norm over channels.

```python
from neural.layers import PixelNorm

norm = PixelNorm()
out = norm(x)  # (N, C, H, W) -> (N, C, H, W), each channel normalized
```

Stabilizes GAN training and is common in progressive growing architectures.

---

#### AdaINLayer

Adaptive Instance Normalization. Normalizes the content tensor by its own mean/std, then reparameterizes using the mean/std of a style tensor.

```python
from neural.layers import AdaINLayer

adain = AdaINLayer()
content = torch.randn(32, 64, 28, 28)
style = torch.randn(32, 64, 28, 28)
out = adain((content, style))  # (N, 64, 28, 28)
```

---

#### MeanStdNorm

Learnable mean/std normalization per channel. Useful when normalization statistics should adapt during training.

```python
from neural.layers import MeanStdNorm

norm = MeanStdNorm(channels=64)  # mu and std are learnable Parameters
out = norm(x)  # (N, 64, H, W) -> (N, 64, H, W)
```

---

#### NormalizationStack2d

Composable normalization stack. Chains BatchNorm, PixelNorm, and InstanceNorm in sequence.

```python
from neural.layers import NormalizationStack2d

stack = NormalizationStack2d(
    norm_types=["batch", "pixel"],
    num_features=64
)
# Equivalent to: nn.BatchNorm2d(64) -> PixelNorm()
out = stack(x)
```

Supported norm types: `"batch"`, `"batchnorm"`, `"pixel"`, `"instance"`, `"instancenorm"`.

---

### Residual Layers

#### SimpleResidualBlock

Two convolutions with identity skip connection.

```python
from neural.layers import SimpleResidualBlock

block = SimpleResidualBlock(64, kernel_size=3)
out = block(x)  # (N, 64, H, W) -> (N, 64, H, W)
```

Optional `activation` parameter for post-convolution activation.

---

#### ResidualBlock

SRGAN-style residual block with BatchNorm between conv and activation.

```python
from neural.layers import ResidualBlock

block = ResidualBlock(
    channels=64, residual_channels=64,
    conv=nn.Conv2d, norm="batch", activation="prelu"
)
out = block(x)
```

Order: `conv -> norm -> activation -> conv -> norm -> + x`

---

#### BottleneckResidualBlock

Residual block with 1x1 channel reduction/expansion for efficiency.

```python
from neural.layers import BottleneckResidualBlock

block = BottleneckResidualBlock(
    channels=64, residual_channels=32, expansion=2
)
out = block(x)
```

Channels flow: `C -> C/r -> C` through the bottleneck.

---

#### ResidualStack

Chains N identical residual blocks with optional input projection (ResNet style).

```python
from neural.layers import ResidualStack

# 3 SRGAN-style blocks, 64 channels throughout
stack = ResidualStack(64, n_blocks=3)
out = stack(x)  # (N, 64, H, W) -> (N, 64, H, W)

# Input has 128 channels -> projected to 64
stack = ResidualStack(residual_channels=64, in_channels=128)
out = stack(x)  # (N, 128, H, W) -> (N, 64, H, W)
```

Supports `block_type`: `"srgan"`, `"preact"`, `"bottleneck"`.

---

### Pooling Layers

#### AttentionPooling

Attention-based pooling with a learnable query vector for transformer use cases.

```python
from neural.layers import AttentionPooling

pool = AttentionPooling(embed_dim=128, num_heads=4, dropout=0.1)
x = torch.randn(32, 20, 128)
mask = torch.ones(32, 20, dtype=torch.bool)
mask[:, 15:] = False
out = pool(x, mask)  # (32, 128)
```

Handles masking to ignore padding positions. Uses multi-head attention with Xavier initialization.

---

#### MaskedAveragePooling

Masked average pooling over sequence dimension, ignoring masked positions.

```python
from neural.layers import MaskedAveragePooling

pool = MaskedAveragePooling(dim=1, eps=1e-8)
x = torch.randn(32, 20, 128)
mask = torch.ones(32, 20, dtype=torch.bool)
mask[:, 15:] = False
out = pool(x, mask)  # (32, 128)
```

---

#### MaskedMaxPooling

Masked max pooling over sequence dimension, ignoring masked positions.

```python
from neural.layers import MaskedMaxPooling

pool = MaskedMaxPooling(dim=1)
x = torch.randn(32, 20, 128)
mask = torch.ones(32, 20, dtype=torch.bool)
mask[:, 15:] = False
out = pool(x, mask)  # (32, 128)
```

---

#### CrossChannelPooling2d

Pools over the channel dimension at each spatial position (opposite of standard pooling).

```python
from neural.layers import CrossChannelPooling2d

# Basic (uniform averaging)
pool = CrossChannelPooling2d()
x = torch.randn(4, 64, 28, 28)
out = pool(x)  # (4, 2, 1, 1) [mean_channels, std_channels]

# Learnable weights
pool = CrossChannelPooling2d(channels=64, learnable=True)
out = pool(x)  # (4, 2, 1, 1)
```

Alias: `CrossChannelPool2d`.

---

### Activation Layers

#### Scaling

Learnable per-channel scaling: `y = scale * x`.

```python
from neural.layers import Scaling

scale = Scaling(channels=64, init=1.0)
out = scale(x)
```

---

#### Mish

Smooth non-monotonic activation: `x * tanh(softplus(x))`.

```python
from neural.layers import Mish

act = Mish()
out = act(x)  # element-wise
```

---

#### Swish

Generalized SiLU: `x * sigmoid(beta * x)` with learnable beta.

```python
from neural.layers import Swish

# Learnable beta (default)
swish = Swish(learnable=True)

# Fixed beta (SiLU)
silu = Swish(learnable=False, beta=1.0)
out = silu(x)
```

---

#### GELU

Gaussian Error Linear Unit with tanh approximation.

```python
from neural.layers import GELU

act = GELU()
out = act(x)
```

---

#### GLU

Gated Linear Unit: splits input along `dim` and applies `first_half * sigmoid(second_half)`.

```python
from neural.layers import GLU

glu = GLU(dim=1)
# x: (N, 2*C, H, W) -> out: (N, C, H, W)
out = glu(x)
```

---

#### PReLU

Parametric ReLU: `max(x, 0) + alpha * min(x, 0)`.

```python
from neural.layers import PReLU

# Shared alpha
prelu = PReLU(num_params=1, init=0.25)

# Per-channel alpha
prelu = PReLU(num_params=64)
out = prelu(x)
```

---

#### SiLU

Sigmoid Linear Unit: `x * sigmoid(x)`. Same as Swish with beta=1.

```python
from neural.layers import SiLU

act = SiLU()
out = act(x)
```

---

#### Activation

Factory activation layer. Dispatches to a named activation by string (case-insensitive).

```python
from neural.layers import Activation

act = Activation("mish")
act = Activation("swish", learnable=False, beta=1.0)
act = Activation("glu", dim=1)
act = Activation("prelu", num_params=64)
act = Activation("leakyrelu", alpha=0.01)
out = act(x)
```

Supported types: `relu`, `leakyrelu`, `elu`, `celu`, `mish`, `swish`, `silu`, `gelu`, `glu`, `prelu`, `tanh`, `sigmoid`, `hardtanh`, `hardswish`. Per-type kwargs documented in the class docstring.

---

### Temporal Layers

#### PositionalEncoding

Sinusoidal positional encoding from "Attention is All You Need."

```python
from neural.layers import PositionalEncoding

pe = PositionalEncoding(128, max_len=5000)
x = torch.randn(4, 20, 128)
out = pe(x)  # (4, 20, 128) + positional encoding
```

---

#### LearnedPositionalEncoding

Learnable position embeddings added to input.

```python
from neural.layers import LearnedPositionalEncoding

pe = LearnedPositionalEncoding(128, max_len=5000)
out = pe(x)
```

---

#### Time2Vec

Temporal feature transformation: mix of linear and sinusoidal components.

```python
from neural.layers import Time2Vec

t2v = Time2Vec(input_dim=1, output_dim=64)
# t: (batch, seq_len, 1) -> out: (batch, seq_len, 64)
out = t2v(t)
```

First component is linear (`w*t + phi`), remaining components are sinusoidal (`sin(s*t + phi)`).

---

#### CausalConv1d

1-D causal (left-padded) convolution for autoregressive models.

```python
from neural.layers import CausalConv1d

conv = CausalConv1d(64, 128, kernel_size=3, dilation=1)
x = torch.randn(4, 64, T)
out = conv(x)  # (4, 128, T) -- no future leakage
```

Supports dilation. Pads only the left (past) side.

---

#### WaveNet

WaveNet residual block with causal dilated convolution and gated activation unit.

```python
from neural.layers import WaveNet

w = WaveNet(channels=128, filters=64, kernel_size=3, dilations=(1, 2, 4))
# x: (N, 128, T)
residual, skip = w(x)
# residual: (N, 128, T) -- added to input
# skip: (N, 192, T) -- concatenated skip connections (3 * 64)
```

Each dilation branch applies causal conv + tanh/sigmoid gate + 1x1 skip conv. Skip connections are concatenated across branches; a 1x1 conv maps to the residual width.

---

#### RotaryPositionalEncoding

Rotary positional encoding (RoPE) from PaLM/GPT-J. Rotates query and key vectors by learned angles at each position.

```python
from neural.layers import RotaryPositionalEncoding

rope = RotaryPositionalEncoding(head_dim=64, base=10000.0)
q = torch.randn(2, 4, 10, 64)  # (batch, heads, seq_len, head_dim)
k = torch.randn(2, 4, 10, 64)
q_rot, k_rot = rope(q, k)  # rotated, norm-preserving
```

Includes caching for efficient inference.

---

### Backbone Layers

#### PretrainedBackbone

Wraps torchvision models as feature extractors with freeze/finetune control.

```python
from neural.layers import PretrainedBackbone

# Load pretrained ResNet50, freeze backbone, cut at layer4
backbone = PretrainedBackbone(
    "resnet50", pretrained=True, freeze=True,
    output_layer="layer4", in_channels=3
)
x = torch.randn(4, 3, 224, 224)
out = backbone(x)  # (4, 2048, 7, 7)

# Unfreeze for fine-tuning
backbone.unfreeze()

# Freeze only conv layers (keep BN stats trainable)
backbone.freeze_pretrained_layers()

# Supported models: resnet18/34/50/101/152, resnext*, wide_resnet*,
# efficientnet_b0-b7, efficientnet_v2_s/m/l, mobilenet_v2/v3_small/large,
# densenet121/161/169/201, vgg11/13/16/19 (+bn), convnext_tiny/small/base/large,
# swin_t/s/b (+v2 variants)
```

Key methods: `freeze()`, `unfreeze()`, `freeze_pretrained_layers()`, `unfreeze_pretrained_layers()`.

---

### Output Heads

#### ClassificationHead

Multi-class classification head with optional intermediate extraction.

```python
from neural.layers import ClassificationHead

head = ClassificationHead(d_input=512, d_hidden=256, n_classes=10)
logits = head(x)              # (B, 10)
embedding = head(x, embedding=True)  # (B, 256)
```

Architecture: `Linear -> activation -> dropout -> Linear(n_classes)`. Supported activations: `relu`, `elu`, `celu`, `gelu`, `sigmoid`, `tanh`, `leakyrelu`, `softplus`, `softsign`. Binary classification is just `n_classes=2`.

---

#### RegressionHead

Regression head with optional output activation.

```python
from neural.layers import RegressionHead

head = RegressionHead(d_input=512, d_hidden=256)
pred = head(x)                # (B, 1)
embedding = head(x, embedding=True)  # (B, 256)
```

Architecture: `Linear -> activation -> dropout -> Linear(1)`. Supported activations: `relu`, `elu`, `celu`, `gelu`, `sigmoid`, `tanh`, `leakyrelu`, `softplus`, `softsign`.

---

## Datasets

Two families of iterable datasets for PyTorch DataLoader integration: **Iterable** (single-pass, fixed-length) and **Streaming** (infinite cycle). Both support deterministic shuffling and automatic multi-worker data partitioning.

### Iterable Datasets

#### SingleWorkerIterableDataset

Single-pass iteration over fixed data. One iteration = one full pass, then stops.

```python
from neural.datasets import SingleWorkerIterableDataset
from torch.utils.data import DataLoader

ds = SingleWorkerIterableDataset([1, 2, 3, 4, 5])
for batch in DataLoader(ds, batch_size=2):
    process(batch)  # [1,2] then [3,4] then [5]
```

Override `preprocess(point)` to transform samples; return `None` to filter.

#### ShuffledIterableDataset

Same as SingleWorkerIterableDataset with optional deterministic shuffling. Seed auto-increments per iteration for reproducible but varied shuffles.

```python
from neural.datasets import ShuffledIterableDataset

ds = ShuffledIterableDataset(data, shuffle=True, seed=42)
# First iteration: seed=42 -> 43
# Second iteration: seed=43 -> 44
```

#### MultiWorkerIterableDataset

Auto-partitions data across DataLoader workers via strided slicing. Worker 0 gets indices 0,4,8..., worker 1 gets 1,5,9..., etc.

```python
from neural.datasets import MultiWorkerIterableDataset

ds = MultiWorkerIterableDataset(data, shuffle=True, seed=0)
loader = DataLoader(ds, batch_size=32, num_workers=4)
# Each worker gets a disjoint strided slice
```

### Streaming Datasets

#### SingleWorkerStreamDataset

Infinite stream via `while True: yield from` pattern. Default `preprocess` extracts (x, y) and converts to tensors.

```python
from neural.datasets import SingleWorkerStreamDataset

ds = SingleWorkerStreamDataset([(x1, y1), (x2, y2), ...])
for x, y in DataLoader(ds, batch_size=32):
    train_step(x, y)  # runs forever
```

#### ShuffledStreamDataset

Same as SingleWorkerStreamDataset with deterministic shuffling. Seed auto-increments each cycle.

#### MultiWorkerStreamDataset

Same as ShuffledStreamDataset with automatic multi-worker partitioning via `itertools.islice`.

```python
from neural.datasets import MultiWorkerStreamDataset

ds = MultiWorkerStreamDataset([(i, 0) for i in range(100)], shuffle=True, seed=0)
loader = DataLoader(ds, batch_size=32, num_workers=4)
```

[Back to top](#table-of-contents)

---

## Losses

Placeholder. This section will document the loss functions available in `neural.losses`.

[Back to top](#table-of-contents)

---

## Evaluation

Placeholder. This section will document the metric computation utilities available in `neural.evaluation`.

[Back to top](#table-of-contents)

---

## Training

Placeholder. This section will document the training infrastructure available in `neural.training`.

[Back to top](#table-of-contents)

---

## Utils

Placeholder. This section will document the training utilities available in `neural.utils`.

[Back to top](#table-of-contents)
