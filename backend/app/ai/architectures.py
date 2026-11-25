"""Five custom-trained CNN architectures matching the saved .pth checkpoints."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. ResNet-18 (matches resnet18-new.ipynb / ResNet18)
# ==============================================================================
class BasicBlock(nn.Module):
    """Standard two-layer residual block."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class _ResNet18Base(nn.Module):
    """Shared ResNet-18 backbone parameterized by block class (BasicBlock or SEResNetBlock)."""

    def __init__(self, block_cls: type[nn.Module], num_classes: int = 37, in_channels: int = 3):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block_cls, 64, 2, stride=1)
        self.layer2 = self._make_layer(block_cls, 128, 2, stride=2)
        self.layer3 = self._make_layer(block_cls, 256, 2, stride=2)
        self.layer4 = self._make_layer(block_cls, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block_cls.expansion, num_classes)

    def _make_layer(self, block_cls: type[nn.Module], planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block_cls(self.in_planes, planes, s))
            self.in_planes = planes * block_cls.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.linear(out)


class ResNet18(_ResNet18Base):
    """ResNet-18 matching resnet18_augmented_best.pth."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3):
        super().__init__(BasicBlock, num_classes=num_classes, in_channels=in_channels)


# ==============================================================================
# 2. CustomSEResNet (matches fg-seresnet18-newdataset-al.ipynb — uses .linear head)
# ==============================================================================
class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention block for SEResNet-18."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        weight = self.fc(x).view(b, c, 1, 1)
        return x * weight


class SEResNetBlock(nn.Module):
    """Residual block with Squeeze-and-Excitation attention."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SEBlock(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        return F.relu(out)


class CustomSEResNet(_ResNet18Base):
    """SEResNet-18 matching seresnet18_augmented_best.pth (with .linear classifier)."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3):
        super().__init__(SEResNetBlock, num_classes=num_classes, in_channels=in_channels)


# ==============================================================================
# 3. DenseNet-121 (matches 06_densenet121_new_dataset (1).ipynb / DenseNet121)
# ==============================================================================
class _DenseLayer(nn.Module):
    def __init__(self, in_features: int, growth_rate: int, bn_size: int = 4):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_features)
        self.conv1 = nn.Conv2d(in_features, bn_size * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return torch.cat([x, out], 1)


class _DenseBlock(nn.Module):
    def __init__(self, num_layers: int, in_features: int, growth_rate: int, bn_size: int = 4):
        super().__init__()
        layers = [
            _DenseLayer(in_features + i * growth_rate, growth_rate, bn_size)
            for i in range(num_layers)
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Transition(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_features)
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(F.relu(self.bn(x)))
        return self.pool(out)


class DenseNet121(nn.Module):
    """DenseNet-121 matching densenet121_augmented_best.pth."""

    def __init__(
        self,
        growth_rate: int = 32,
        block_config: tuple[int, ...] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        num_classes: int = 37,
        in_channels: int = 3,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, in_features=num_features, growth_rate=growth_rate, bn_size=bn_size)
            self.features.add_module(f"denseblock{i + 1}", block)
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(in_features=num_features, out_features=num_features // 2)
                self.features.add_module(f"transition{i + 1}", trans)
                num_features //= 2

        self.features.add_module("norm5", nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.features(x))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return self.classifier(out)


# ==============================================================================
# 4 & 5. EfficientNet-B0 & Dual-Stream EfficientNet-B0
# ==============================================================================
class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block with configurable FC bias for B0 vs Dual-Stream."""

    def __init__(self, in_channels: int, se_channels: int, bias: bool = False):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, se_channels, bias=bias),
            nn.SiLU(inplace=True),
            nn.Linear(se_channels, in_channels, bias=bias),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.fc(x).view(b, c, 1, 1)
        return x * w


class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Convolution with Squeeze-and-Excitation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expand_ratio: int = 6,
        se_ratio: float = 0.25,
        se_bias: bool = False,
    ):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        hidden_dim = in_channels * expand_ratio

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True),
            ])

        padding = (kernel_size - 1) // 2
        layers.extend([
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            SqueezeExcitation(hidden_dim, max(1, int(in_channels * se_ratio)), bias=se_bias),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


def _build_efficientnet_b0_blocks(se_bias: bool) -> tuple[nn.Sequential, int]:
    """Construct Tan & Le 2019 Table 1 MBConv stage sequence."""
    b0_configs = [
        (1, 16, 1, 1, 3),
        (6, 24, 2, 2, 3),
        (6, 40, 2, 2, 5),
        (6, 80, 3, 2, 3),
        (6, 112, 3, 1, 5),
        (6, 192, 4, 2, 5),
        (6, 320, 1, 1, 3),
    ]
    blocks = []
    c_in = 32
    for expand_ratio, c_out, num_layers, first_stride, kernel_size in b0_configs:
        for i in range(num_layers):
            stride = first_stride if i == 0 else 1
            blocks.append(
                MBConv(
                    c_in,
                    c_out,
                    kernel_size=kernel_size,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    se_ratio=0.25,
                    se_bias=se_bias,
                )
            )
            c_in = c_out
    return nn.Sequential(*blocks), c_in


class EfficientNetB0(nn.Module):
    """EfficientNet-B0 matching efficientnet_b0_augmented_best.pth."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3, drop_rate: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )
        self.blocks, c_in = _build_efficientnet_b0_blocks(se_bias=False)
        self.head = nn.Sequential(
            nn.Conv2d(c_in, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(drop_rate),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


class DualStreamEfficientNet(nn.Module):
    """Dual-Stream EfficientNet-B0 matching dual_stream_efficientnet_best.pth."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3, drop_rate: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )
        self.blocks, c_in = _build_efficientnet_b0_blocks(se_bias=True)
        self.head_conv = nn.Sequential(
            nn.Conv2d(c_in, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.gate = nn.Sequential(
            nn.Linear(1280 * 2, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, 1280),
            nn.Sigmoid(),
        )
        self.ln = nn.LayerNorm(1280)
        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(1280, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.blocks(h)
        h = self.head_conv(h)
        h = self.pool(h)
        return self.flatten(h)

    def forward(self, x_body: torch.Tensor, x_head: torch.Tensor | None = None) -> torch.Tensor:
        feat_body = self.extract_features(x_body)
        if x_head is None:
            return self.classifier(feat_body)
        feat_head = self.extract_features(x_head)
        cat = torch.cat([feat_body, feat_head], dim=1)
        g = self.gate(cat)
        fused = self.ln(feat_body + g * feat_head)
        return self.classifier(fused)
