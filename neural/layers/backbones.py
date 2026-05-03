"""
Pretrained backbone loaders for transfer learning.

Wraps torchvision models as feature extractors with freeze/finetune control.
Supports ResNet, EfficientNet, MobileNet, and other torchvision families.
"""

from typing import Literal

import torch
import torch.nn as nn
import torchvision.models as models


_MODEL_REGISTRY = {
    # ResNet families
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
    # ResNet variants
    "resnext50_32x4d": models.resnext50_32x4d,
    "resnext101_32x8d": models.resnext101_32x8d,
    "wide_resnet50_2": models.wide_resnet50_2,
    "wide_resnet101_2": models.wide_resnet101_2,
    # EfficientNet families
    "efficientnet_b0": models.efficientnet_b0,
    "efficientnet_b1": models.efficientnet_b1,
    "efficientnet_b2": models.efficientnet_b2,
    "efficientnet_b3": models.efficientnet_b3,
    "efficientnet_b4": models.efficientnet_b4,
    "efficientnet_b5": models.efficientnet_b5,
    "efficientnet_b6": models.efficientnet_b6,
    "efficientnet_b7": models.efficientnet_b7,
    "efficientnet_v2_s": models.efficientnet_v2_s,
    "efficientnet_v2_m": models.efficientnet_v2_m,
    "efficientnet_v2_l": models.efficientnet_v2_l,
    # MobileNet families
    "mobilenet_v2": models.mobilenet_v2,
    "mobilenet_v3_small": models.mobilenet_v3_small,
    "mobilenet_v3_large": models.mobilenet_v3_large,
    # Densenet families
    "densenet121": models.densenet121,
    "densenet169": models.densenet169,
    "densenet161": models.densenet161,
    "densenet201": models.densenet201,
    # VGG
    "vgg11": models.vgg11,
    "vgg13": models.vgg13,
    "vgg16": models.vgg16,
    "vgg19": models.vgg19,
    "vgg11_bn": models.vgg11_bn,
    "vgg13_bn": models.vgg13_bn,
    "vgg16_bn": models.vgg16_bn,
    "vgg19_bn": models.vgg19_bn,
    # ConvNeXt
    "convnext_tiny": models.convnext_tiny,
    "convnext_small": models.convnext_small,
    "convnext_base": models.convnext_base,
    "convnext_large": models.convnext_large,
    # Swin Transformer
    "swin_t": models.swin_t,
    "swin_s": models.swin_s,
    "swin_b": models.swin_b,
    "swin_v2_t": models.swin_v2_t,
    "swin_v2_s": models.swin_v2_s,
    "swin_v2_b": models.swin_v2_b,
}


_OUTPUT_LAYERS = {
    "resnet18": ["layer1", "layer2", "layer3", "layer4", "avgpool", "fc"],
    "resnet34": ["layer1", "layer2", "layer3", "layer4", "avgpool", "fc"],
    "resnet50": ["layer1", "layer2", "layer3", "layer4", "avgpool", "fc"],
    "resnet101": ["layer1", "layer2", "layer3", "layer4", "avgpool", "fc"],
    "resnet152": ["layer1", "layer2", "layer3", "layer4", "avgpool", "fc"],
    "resnext50_32x4d": ["layer1", "layer2", "layer3", "layer4", "avgpool", "fc"],
    "resnext101_32x8d": ["layer1", "layer2", "layer3", "layer4", "avgpool", "fc"],
    "wide_resnet50_2": ["layer1", "layer2", "layer3", "layer4", "avgpool", "fc"],
    "wide_resnet101_2": ["layer1", "layer2", "layer3", "layer4", "avgpool", "fc"],
    "efficientnet_b0": ["features", "avgpool", "classifier"],
    "efficientnet_b1": ["features", "avgpool", "classifier"],
    "efficientnet_b2": ["features", "avgpool", "classifier"],
    "efficientnet_b3": ["features", "avgpool", "classifier"],
    "efficientnet_b4": ["features", "avgpool", "classifier"],
    "efficientnet_b5": ["features", "avgpool", "classifier"],
    "efficientnet_b6": ["features", "avgpool", "classifier"],
    "efficientnet_b7": ["features", "avgpool", "classifier"],
    "efficientnet_v2_s": ["features", "avgpool", "classifier"],
    "efficientnet_v2_m": ["features", "avgpool", "classifier"],
    "efficientnet_v2_l": ["features", "avgpool", "classifier"],
    "mobilenet_v2": ["features", "avgpool", "classifier"],
    "mobilenet_v3_small": ["features", "avgpool", "classifier"],
    "mobilenet_v3_large": ["features", "avgpool", "classifier"],
    "densenet121": ["features", "classifier"],
    "densenet169": ["features", "classifier"],
    "densenet161": ["features", "classifier"],
    "densenet201": ["features", "classifier"],
    "vgg11": ["features", "avgpool", "classifier"],
    "vgg13": ["features", "avgpool", "classifier"],
    "vgg16": ["features", "avgpool", "classifier"],
    "vgg19": ["features", "avgpool", "classifier"],
    "vgg11_bn": ["features", "avgpool", "classifier"],
    "vgg13_bn": ["features", "avgpool", "classifier"],
    "vgg16_bn": ["features", "avgpool", "classifier"],
    "vgg19_bn": ["features", "avgpool", "classifier"],
    "convnext_tiny": ["features", "avgpool", "classifier"],
    "convnext_small": ["features", "avgpool", "classifier"],
    "convnext_base": ["features", "avgpool", "classifier"],
    "convnext_large": ["features", "avgpool", "classifier"],
}


class PretrainedBackbone(nn.Module):
    """Load a pretrained torchvision model as a feature extractor.

    Wraps any torchvision model family from ``torchvision.models`` and
    exposes it as a sequence of layers with freeze/finetune control.

    Parameters:
        name: model family name (e.g. ``"resnet50"``, ``"efficientnet_b0"``).
        pretrained: load ImageNet weights (default True).
        freeze: freeze all parameters (default False).
        output_layer: which stage to cut the output from.
            Choose from ``"layer1"``-``"layer4"`` or ``"avgpool"`` for ResNet.
            Choose from ``"features"`` or ``"avgpool"`` for EfficientNet/MobileNet.
        in_channels: override input channels (replaces first conv if needed).
            Only 1 and 3 are handled automatically.
        aux_heads: number of extra classification heads for multi-task learning.

    Example::

        backbone = PretrainedBackbone(
            "resnet50", pretrained=True, freeze=True,
            output_layer="layer4", in_channels=1
        )
        # x: (N, 1, 224, 224) -> out: (N, 2048, 7, 7)
        out = backbone(x)
    """

    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        freeze: bool = False,
        output_layer: str = "avgpool",
        in_channels: int = 3,
        aux_heads: int = 0,
    ) -> None:
        super().__init__()

        if name not in _MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model: {name!r}. "
                f"Choose from: {', '.join(sorted(_MODEL_REGISTRY))}"
            )

        if output_layer not in _OUTPUT_LAYERS.get(name, []):
            raise ValueError(
                f"Unknown output_layer {output_layer!r} for {name}. "
                f"Choose from: {', '.join(_OUTPUT_LAYERS.get(name, []))}"
            )

        self.name = name
        self._output_layer = output_layer
        self.in_channels = in_channels

        # Load the model
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT if "resnet18" in name else None
            # Auto-select default weights for the model
            model = _MODEL_REGISTRY[name](weights="IMAGENET1K_V1" if pretrained else None)
        else:
            model = _MODEL_REGISTRY[name](weights=None)

        # Handle in_channels override (replaces first conv layer)
        if in_channels != 3:
            model = self._replace_first_conv(model, in_channels)

        # Freeze if requested
        if freeze:
            for param in model.parameters():
                param.requires_grad = False

        self.model = model
        self.frozen = freeze
        self.num_channels = self._get_output_channels(model, output_layer)

        # Auxiliary classification heads for multi-task learning
        self.aux_heads = nn.ModuleList()
        for _ in range(aux_heads):
            self.aux_heads.append(nn.Linear(self.num_channels, 1000))

    def freeze(self) -> None:
        """Freeze all model parameters (no gradient computation)."""
        for param in self.model.parameters():
            param.requires_grad = False
        self.frozen = True

    def unfreeze(self) -> None:
        """Unfreeze all model parameters (trainable with gradients)."""
        for param in self.model.parameters():
            param.requires_grad = True
        self.frozen = False

    def freeze_pretrained_layers(self) -> None:
        """Freeze pretrained conv/dense layers while keeping BN layers trainable.

        BatchNorm layers need running statistics to update during training.
        Freezing only the feature-extraction layers preserves this behavior.
        """
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                continue
            for param in module.parameters():
                param.requires_grad = False
        # Ensure BN parameters stay trainable (parent module.parameters() includes them)
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                for param in module.parameters():
                    param.requires_grad = True
        self.frozen = True

    def unfreeze_pretrained_layers(self) -> None:
        """Unfreeze pretrained conv/dense layers while keeping BN layers trainable.

        Only unfreezes feature-extraction layers; BN layers are left untouched
        since they were never frozen.
        """
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                continue
            for param in module.parameters():
                param.requires_grad = True
        self.frozen = False

    @property
    def output_layer(self) -> str:
        return self._output_layer

    @output_layer.setter
    def output_layer(self, value: str) -> None:
        if value not in _OUTPUT_LAYERS.get(self.name, []):
            raise ValueError(
                f"Unknown output_layer {value!r} for {self.name!r}. "
                f"Choose from: {', '.join(_OUTPUT_LAYERS.get(self.name, []))}"
            )
        self._output_layer = value

    @staticmethod
    def _replace_first_conv(model: nn.Module, in_channels: int) -> nn.Module:
        """Replace first conv layer to accept different number of input channels."""
        if isinstance(model, models.ResNet):
            old_conv = model.conv1
            model.conv1 = nn.Conv2d(
                in_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                dilation=old_conv.dilation,
                groups=old_conv.groups,
                bias=old_conv.bias is not None,
                padding_mode=old_conv.padding_mode,
            )
        elif isinstance(model, models.DenseNet):
            old_conv = model.features.conv0
            model.features.conv0 = nn.Conv2d(
                in_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                dilation=old_conv.dilation,
                groups=old_conv.groups,
                bias=old_conv.bias is not None,
                padding_mode=old_conv.padding_mode,
            )
        elif isinstance(model, (models.EfficientNet, models.MobileNetV2)):
            old_conv = model.features[0][0]
            model.features[0][0] = nn.Conv2d(
                in_channels, old_conv.out_channels, old_conv.kernel_size,
                old_conv.stride, old_conv.padding, old_conv.dilation,
                old_conv.groups, old_conv.bias is not None, old_conv.padding_mode,
            )
        elif isinstance(model, models.VGG):
            old_conv = model.features[0]
            model.features[0] = nn.Conv2d(
                in_channels, old_conv.out_channels, old_conv.kernel_size,
                old_conv.stride, old_conv.padding, old_conv.dilation,
                old_conv.bias is not None,
            )
        elif isinstance(model, models.ConvNeXt):
            old_conv = model.features[0]
            model.features[0] = nn.Conv2d(
                in_channels, old_conv.out_channels, old_conv.kernel_size,
                old_conv.stride, old_conv.padding, old_conv.dilation,
                old_conv.bias is not None,
            )
        elif isinstance(model, models.SwinTransformer):
            old_conv = model.features[0].stem
            new_patch = nn.Sequential(
                nn.Conv2d(in_channels, 96, kernel_size=4, stride=4),
                nn.LayerNorm(96),
            )
            model.features[0].stem = new_patch
        else:
            raise ValueError(f"Cannot replace first conv for model type: {type(model)}")
        return model

    @staticmethod
    def _get_output_channels(model: nn.Module, output_layer: str) -> int:
        """Determine output channels for the specified output_layer."""
        if output_layer == "fc":
            return model.fc.in_features
        elif output_layer == "classifier":
            return model.classifier.in_features
        elif output_layer == "avgpool":
            if isinstance(model, models.ResNet):
                last = model.layer4[-1]
                # ResNet-18/34 use BasicBlock (conv1+conv2), ResNet-50+ use Bottleneck (conv1+conv2+conv3)
                if "BasicBlock" in type(last).__name__:
                    return last.conv2.out_channels
                else:
                    return last.conv2.out_channels
            elif isinstance(model, models.EfficientNet):
                return model.classifier[-1].in_features
            elif isinstance(model, models.MobileNetV2):
                return model.classifier[-1].in_features
            elif isinstance(model, models.VGG):
                return model.classifier[1].in_features
            elif isinstance(model, models.DenseNet):
                return model.classifier.in_features
            elif isinstance(model, models.ConvNeXt):
                return model.classifier.in_features
            return 512
        elif output_layer == "features":
            if isinstance(model, models.EfficientNet):
                return model.classifier[1].in_features
            elif isinstance(model, models.MobileNetV2):
                return model.classifier[-1].in_features
            elif isinstance(model, models.VGG):
                return model.classifier[1].in_features
            elif isinstance(model, models.DenseNet):
                return model.classifier.in_features
            elif isinstance(model, models.ConvNeXt):
                return model.classifier.in_features
            return 512
        elif output_layer == "layer1":
            if isinstance(model, models.ResNet):
                return model.layer1[-1].conv2.out_channels
        elif output_layer == "layer2":
            if isinstance(model, models.ResNet):
                return model.layer2[-1].conv2.out_channels
        elif output_layer == "layer3":
            if isinstance(model, models.ResNet):
                return model.layer3[-1].conv2.out_channels
        elif output_layer == "layer4":
            if isinstance(model, models.ResNet):
                last = model.layer4[-1]
                # ResNet-18/34 use BasicBlock, ResNet-50+ use Bottleneck
                if type(last).__name__ == "BasicBlock":
                    return last.conv2.out_channels
                else:
                    return last.conv2.out_channels
        return 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the backbone.

        Args:
            x: input tensor.

        Returns:
            Feature map at the specified output_layer.
        """
        if isinstance(self.model, models.ResNet):
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            out = {"layer1": self.model.layer1(x)}
            x = out["layer1"]
            out["layer2"] = self.model.layer2(x)
            x = out["layer2"]
            out["layer3"] = self.model.layer3(x)
            x = out["layer3"]
            out["layer4"] = self.model.layer4(x)
            x = out["layer4"]

            out["avgpool"] = self.model.avgpool(x).flatten(1)
            out["fc"] = out["avgpool"]
            return out[self._output_layer]

        elif isinstance(self.model, models.EfficientNet):
            feats = self.model.features(x)
            out = {"features": feats, "avgpool": self.model.avgpool(feats).flatten(1)}
            out["classifier"] = out["avgpool"]
            return out[self._output_layer]

        elif isinstance(self.model, models.MobileNetV2):
            feats = self.model.features(x)
            out = {"features": feats, "avgpool": self.model.avgpool(feats).flatten(1)}
            out["classifier"] = out["avgpool"]
            return out[self._output_layer]

        elif isinstance(self.model, models.DenseNet):
            feats = self.model.features(x)
            feats = self.model.norm5(feats)
            feats = torch.relu(feats)
            feats_flat = torch.flatten(feats, 1)
            out = {"features": feats_flat, "classifier": self.model.classifier(feats_flat)}
            return out[self._output_layer]

        elif isinstance(self.model, models.VGG):
            feats = self.model.features(x)
            out = {"features": feats, "avgpool": self.model.avgpool(feats).flatten(1)}
            out["classifier"] = out["avgpool"]
            return out[self._output_layer]

        elif isinstance(self.model, models.ConvNeXt):
            feats = self.model.features(x)
            out = {"features": feats, "avgpool": self.model.avgpool(feats).flatten(1)}
            out["classifier"] = out["avgpool"]
            return out[self._output_layer]

        elif isinstance(self.model, models.SwinTransformer):
            # Swin outputs [B, H, W, C] -> flatten to [B, C]
            feats = self.model.forward_features(x)
            out = {"features": feats}
            out["avgpool"] = feats.mean(dim=[2, 3])
            out["classifier"] = self.model.head(out["avgpool"])
            return out[self._output_layer]

        raise NotImplementedError(f"Forward not implemented for model type: {type(self.model)}")
