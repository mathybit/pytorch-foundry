# neural.layers package - custom PyTorch layers

# Convolutional layers
from neural.layers.convolutional import MultiDilationConv1d
from neural.layers.convolutional import MultiDilationConv2d
from neural.layers.convolutional import MultiDilationConv3d
from neural.layers.convolutional import SubPixelConv2d

# Normalization layers
from neural.layers.normalization import PixelNorm
from neural.layers.normalization import AdaINLayer
from neural.layers.normalization import MeanStdNorm
from neural.layers.normalization import NormalizationStack2d

# Residual layers
from neural.layers.residual import SimpleResidualBlock
from neural.layers.residual import ResidualBlock
from neural.layers.residual import BottleneckResidualBlock
from neural.layers.residual import ResidualStack

# Pooling layers
from neural.layers.pooling import AttentionPooling
from neural.layers.pooling import MaskedAveragePooling
from neural.layers.pooling import MaskedMaxPooling
from neural.layers.pooling import CrossChannelPooling2d
from neural.layers.pooling import CrossChannelPool2d

# Activation layers
from neural.layers.activations import Scaling
from neural.layers.activations import Mish
from neural.layers.activations import Swish
from neural.layers.activations import GELU
from neural.layers.activations import GLU
from neural.layers.activations import PReLU
from neural.layers.activations import SiLU
from neural.layers.activations import Activation

# Temporal layers
from neural.layers.temporal import PositionalEncoding
from neural.layers.temporal import LearnedPositionalEncoding
from neural.layers.temporal import Time2Vec
from neural.layers.temporal import CausalConv1d
from neural.layers.temporal import WaveNet
from neural.layers.temporal import RotaryPositionalEncoding

# Backbone layers
from neural.layers.backbones import PretrainedBackbone

# Output heads
from neural.layers.output_heads import ClassificationHead
from neural.layers.output_heads import RegressionHead

__all__ = [
    # Convolutional layers
    "MultiDilationConv1d",
    "MultiDilationConv2d",
    "MultiDilationConv3d",
    "SubPixelConv2d",
    # Normalization layers
    "PixelNorm",
    "AdaINLayer",
    "MeanStdNorm",
    "NormalizationStack2d",
    # Residual layers
    "SimpleResidualBlock",
    "ResidualBlock",
    "BottleneckResidualBlock",
    "ResidualStack",
    # Pooling layers
    "AttentionPooling",
    "MaskedAveragePooling",
    "MaskedMaxPooling",
    "CrossChannelPooling2d",
    "CrossChannelPool2d",
    # Activation layers
    "Scaling",
    "Mish",
    "Swish",
    "GELU",
    "GLU",
    "PReLU",
    "SiLU",
    "Activation",
    # Temporal layers
    "PositionalEncoding",
    "LearnedPositionalEncoding",
    "Time2Vec",
    "CausalConv1d",
    "WaveNet",
    "RotaryPositionalEncoding",
    # Backbone layers
    "PretrainedBackbone",
    # Output heads
    "ClassificationHead",
    "RegressionHead",
]
