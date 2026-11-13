"""
Pure PyTorch Model Architectures for Oxford-IIIT Pet Classification (100% From Scratch).
NO external pre-packaged model libraries (e.g., torchvision.models) are used.
Architectures implemented from first principles based on R&D/CNN.txt:
1. custom_cnn: 6-Layer Baseline ConvNet
2. alexnet: Modernized AlexNet with BatchNorm
3. vgg11_bn: VGG-11 with Batch Normalization
4. resnet18: ResNet-18 Residual Network
5. resnet34: ResNet-34 Deep Residual Network
6. resnet50: ResNet-50 Deep Bottleneck Residual Network
7. densenet121: DenseNet-121 with Dense Connectivity
8. mobilenet_v2: MobileNetV2 with Inverted Residuals & Linear Bottlenecks
9. seresnet18: SENet (ResNet-18 with Squeeze-and-Excitation Channel Attention)
10. efficientnet: EfficientNet-B0 (Google Research - Tan & Le, 2019)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. Custom 6-Layer Baseline CNN
# ==============================================================================
class CustomCNN(nn.Module):
    def __init__(self, num_classes=37, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # -> 112x112

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # -> 56x56

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # -> 28x28

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # -> 14x14
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.classifier(self.features(x))


# ==============================================================================
# 2. Modernized AlexNet (with BatchNorm)
# ==============================================================================
class AlexNetBN(nn.Module):
    def __init__(self, num_classes=37, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.classifier(self.features(x))


# ==============================================================================
# 3. VGG-11 with BatchNorm (From Scratch)
# ==============================================================================
class VGG11BN(nn.Module):
    def __init__(self, num_classes=37, in_channels=3):
        super().__init__()
        # VGG-11 config: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        layers = []
        c_in = in_channels
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.extend([
                    nn.Conv2d(c_in, v, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True)
                ])
                c_in = v
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.classifier(self.features(x))


# ==============================================================================
# 4. ResNet (ResNet-18, ResNet-34 & ResNet-50 From Scratch)
# ==============================================================================
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetFromScratch(nn.Module):
    def __init__(self, block, num_blocks, num_classes=37, in_channels=3):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self._init_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


def ResNet18(num_classes=37, in_channels=3):
    return ResNetFromScratch(BasicBlock, [2, 2, 2, 2], num_classes, in_channels)


def ResNet34(num_classes=37, in_channels=3):
    return ResNetFromScratch(BasicBlock, [3, 4, 6, 3], num_classes, in_channels)


def ResNet50(num_classes=37, in_channels=3):
    return ResNetFromScratch(Bottleneck, [3, 4, 6, 3], num_classes, in_channels)


# ==============================================================================
# 5. DenseNet-121 (From Scratch)
# ==============================================================================
class _DenseLayer(nn.Module):
    def __init__(self, in_features, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_features)
        self.conv1 = nn.Conv2d(in_features, bn_size * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, in_features, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(_DenseLayer(in_features + i * growth_rate, growth_rate, bn_size, drop_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class _Transition(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_features)
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.pool(out)
        return out


class DenseNet121(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64,
                 bn_size=4, drop_rate=0.0, num_classes=37, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, in_features=num_features,
                                growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
            self.features.add_module(f'denseblock{i + 1}', block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(in_features=num_features, out_features=num_features // 2)
                self.features.add_module(f'transition{i + 1}', trans)
                num_features = num_features // 2

        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


# ==============================================================================
# 6. MobileNetV2 (From Scratch)
# ==============================================================================
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.extend([
                nn.Conv2d(inp, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        # 3x3 depthwise conv
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # 1x1 linear pointwise conv
            nn.Conv2d(hidden_dim, oup, kernel_size=1, bias=False),
            nn.BatchNorm2d(oup)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2FromScratch(nn.Module):
    def __init__(self, num_classes=37, in_channels=3, width_mult=1.0):
        super().__init__()
        input_channel = int(32 * width_mult)
        last_channel = int(1280 * width_mult)
        # [t, c, n, s] -> expand_ratio, out_channel, num_blocks, stride
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # First conv layer
        features = [
            nn.Conv2d(in_channels, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        ]

        # Building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # Building last several layers
        features.extend([
            nn.Conv2d(input_channel, last_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6(inplace=True)
        ])
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.classifier(self.features(x))


# ==============================================================================
# 7. SENet: SEResNet-18 (From Scratch)
# ==============================================================================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        weight = self.fc(x).view(b, c, 1, 1)
        return x * weight


class SEResNetBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
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
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SEResNet18(nn.Module):
    def __init__(self, num_classes=37, in_channels=3):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self._init_weights()

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(SEResNetBlock(self.in_planes, planes, s))
            self.in_planes = planes * SEResNetBlock.expansion
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


# ==============================================================================
# 8. EfficientNet-B0 (Google Research - Tan & Le, 2019 From Scratch)
# ==============================================================================
class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, se_channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, se_channels, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(se_channels, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.fc(x).view(b, c, 1, 1)
        return x * w


class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Convolution with SE Block (Tan & Le, 2019)."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio

        layers = []
        # Expansion phase
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True)
            ])
        else:
            hidden_dim = in_channels

        # Depthwise convolution
        padding = (kernel_size - 1) // 2
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        ])

        # Squeeze-and-Excitation
        if se_ratio > 0:
            se_channels = max(1, int(in_channels * se_ratio))
            layers.append(SqueezeExcitation(hidden_dim, se_channels))

        # Output projection
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class EfficientNetB0(nn.Module):
    """EfficientNet-B0 (Google Research - Tan & Le, 2019 Table 1) 100% From Scratch."""
    def __init__(self, num_classes=37, in_channels=3):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )

        # Tan & Le 2019 Table 1 specifications:
        # [expand_ratio, out_channels, num_layers, stride, kernel_size]
        b0_configs = [
            [1, 16, 1, 1, 3],
            [6, 24, 2, 2, 3],
            [6, 40, 2, 2, 5],
            [6, 80, 3, 2, 3],
            [6, 112, 3, 1, 5],
            [6, 192, 4, 2, 5],
            [6, 320, 1, 1, 3]
        ]

        blocks = []
        c_in = 32
        for t, c_out, num_layers, s, k in b0_configs:
            for i in range(num_layers):
                stride = s if i == 0 else 1
                blocks.append(MBConv(c_in, c_out, kernel_size=k, stride=stride, expand_ratio=t))
                c_in = c_out
        self.blocks = nn.Sequential(*blocks)

        # Head
        self.head = nn.Sequential(
            nn.Conv2d(c_in, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


# Alias for compatibility
EfficientResNet = EfficientNetB0


# ==============================================================================
# Model Factory (All 10 Models from Scratch)
# ==============================================================================

# ==============================================================================
# 11. CLIP From Scratch (Radford et al., ICML 2021)
# Contrastive Language-Image Pre-training: Vision Transformer + Text Transformer
# ==============================================================================

class SimpleTokenizer:
    """
    Standalone deterministic tokenizer for CLIP prompt templates.
    Zero external dependencies, 100% in-vocabulary for Oxford-IIIT Pet breeds.
    """
    def __init__(self, breeds=None):
        self.special_tokens = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
        self.vocab = dict(self.special_tokens)
        corpus = [
            'a photo of a type of pet dog cat animal cute small big furry looking at camera',
            'a close up photo of the face ear paw body tail breed puppy kitten',
            'a portrait of a beautiful friendly sitting standing'
        ]
        if breeds:
            for b in breeds:
                corpus.append(b.replace('_', ' '))
        import re
        words = set()
        for text in corpus:
            words.update(re.findall(r"[a-zA-Z0-9]+", text.lower()))
        for w in sorted(words):
            if w not in self.vocab:
                self.vocab[w] = len(self.vocab)

    def encode(self, text: str, max_length: int = 32) -> torch.Tensor:
        import re
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        ids = [self.special_tokens['<start>']]
        for token in tokens:
            ids.append(self.vocab.get(token, self.special_tokens['<unk>']))
            if len(ids) >= max_length - 1:
                break
        ids.append(self.special_tokens['<end>'])
        while len(ids) < max_length:
            ids.append(self.special_tokens['<pad>'])
        return torch.tensor(ids, dtype=torch.long)


class ResidualAttentionBlock(nn.Module):
    """Multi-Head Self-Attention with Pre-LayerNorm and MLP (Radford et al. 2021)."""
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.ln_2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        norm_x = self.ln_1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x


class CLIPVisionTransformer(nn.Module):
    """Vision Transformer (ViT) Image Encoder for CLIP from scratch."""
    def __init__(self, input_resolution: int = 224, patch_size: int = 16, width: int = 256,
                 layers: int = 6, heads: int = 8, output_dim: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        num_patches = (input_resolution // patch_size) ** 2
        self.positional_embedding = nn.Parameter(scale * torch.randn(num_patches + 1, width))
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = nn.ModuleList([ResidualAttentionBlock(width, heads) for _ in range(layers)])
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        x = self.conv1(x)
        x = x.reshape(B, x.shape[1], -1).permute(0, 2, 1)
        cls_tokens = self.class_embedding.expand(B, 1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.positional_embedding
        x = self.ln_pre(x)
        for block in self.transformer:
            x = block(x)
        x = self.ln_post(x[:, 0, :])
        x = x @ self.proj
        return x


class CLIPTextTransformer(nn.Module):
    """Transformer Text Encoder for CLIP from scratch."""
    def __init__(self, vocab_size: int = 500, max_len: int = 32, width: int = 256,
                 layers: int = 6, heads: int = 8, output_dim: int = 256):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, width)
        scale = width ** -0.5
        self.positional_embedding = nn.Parameter(scale * torch.randn(max_len, width))
        self.transformer = nn.ModuleList([ResidualAttentionBlock(width, heads) for _ in range(layers)])
        self.ln_final = nn.LayerNorm(width)
        self.text_projection = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, text: torch.Tensor):
        x = self.token_embedding(text)
        x = x + self.positional_embedding[:x.shape[1], :]
        for block in self.transformer:
            x = block(x)
        x = self.ln_final(x)
        eos_indices = text.argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), eos_indices] @ self.text_projection
        return x


class CLIPFromScratch(nn.Module):
    """
    OpenAI CLIP replicated 100% from scratch (Radford et al., ICML 2021).
    Supports Contrastive Pre-Training & Zero-Shot Multi-Class Inference.
    """
    def __init__(self, embed_dim: int = 256, num_classes: int = 37, class_names: list = None):
        super().__init__()
        self.num_classes = num_classes
        self.class_names = class_names
        self.tokenizer = SimpleTokenizer(breeds=class_names)
        self.visual = CLIPVisionTransformer(width=256, layers=6, heads=8, output_dim=embed_dim)
        self.text = CLIPTextTransformer(vocab_size=len(self.tokenizer.vocab) + 50, width=256,
                                        layers=6, heads=8, output_dim=embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        self.register_buffer('text_classifier_weights', None)

        if class_names:
            self.build_classifier_weights(class_names)

    def encode_image(self, image: torch.Tensor):
        features = self.visual(image)
        return F.normalize(features, dim=-1)

    def encode_text(self, text: torch.Tensor):
        features = self.text(text)
        return F.normalize(features, dim=-1)

    def build_classifier_weights(self, class_names: list, device='cpu'):
        weights = []
        for name in class_names:
            prompt = f"a photo of a {name.replace('_', ' ')}, a type of pet."
            tokens = self.tokenizer.encode(prompt).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = self.encode_text(tokens)
            weights.append(feat)
        self.text_classifier_weights = torch.cat(weights, dim=0)

    def forward(self, image: torch.Tensor, text: torch.Tensor = None):
        if text is not None:
            im_feat = self.encode_image(image)
            tx_feat = self.encode_text(text)
            scale = self.logit_scale.exp().clamp(max=100)
            logits_i = scale * (im_feat @ tx_feat.t())
            logits_t = logits_i.t()
            return logits_i, logits_t
        else:
            im_feat = self.encode_image(image)
            scale = self.logit_scale.exp().clamp(max=100)
            if self.text_classifier_weights is None:
                # Fallback: identity / dummy weights if not initialized
                return im_feat[:, :self.num_classes]
            logits = scale * (im_feat @ self.text_classifier_weights.to(im_feat.device).t())
            return logits

SUPPORTED_MODELS = [
    "custom_cnn",
    "alexnet",
    "vgg11_bn",
    "resnet18",
    "resnet34",
    "resnet50",
    "densenet121",
    "mobilenet_v2",
    "seresnet18",
    "efficientnet",
    "clip"
]


def build_model(model_name: str, num_classes: int = 37, in_channels: int = 3) -> nn.Module:
    """Builds and returns model instantiated strictly from scratch with Kaiming initialization."""
    name = model_name.lower().strip()
    if name == "custom_cnn":
        return CustomCNN(num_classes=num_classes, in_channels=in_channels)
    elif name == "alexnet":
        return AlexNetBN(num_classes=num_classes, in_channels=in_channels)
    elif name == "vgg11_bn":
        return VGG11BN(num_classes=num_classes, in_channels=in_channels)
    elif name == "resnet18":
        return ResNet18(num_classes=num_classes, in_channels=in_channels)
    elif name == "resnet34":
        return ResNet34(num_classes=num_classes, in_channels=in_channels)
    elif name == "resnet50":
        return ResNet50(num_classes=num_classes, in_channels=in_channels)
    elif name == "densenet121":
        return DenseNet121(num_classes=num_classes, in_channels=in_channels)
    elif name == "mobilenet_v2":
        return MobileNetV2FromScratch(num_classes=num_classes, in_channels=in_channels)
    elif name == "seresnet18":
        return SEResNet18(num_classes=num_classes, in_channels=in_channels)
    elif name in ["efficientnet", "efficientnet_b0", "efficient_resnet"]:
        return EfficientNetB0(num_classes=num_classes, in_channels=in_channels)
    elif name == "clip":
        return CLIPFromScratch(embed_dim=256, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {SUPPORTED_MODELS}")


def count_parameters(model: nn.Module):
    """Returns (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
