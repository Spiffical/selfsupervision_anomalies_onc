import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

try:
    import torchvision.models as tvm
except Exception:
    tvm = None


class SmallCNN(nn.Module):
    def __init__(self, in_ch: int = 1, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
        x = self.dropout(x)
        logits = self.head(x)
        return logits


class DeepCNN(nn.Module):
    def __init__(self, in_ch: int = 1, width: int = 64, depth: int = 6, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        channels = [in_ch] + [width] * depth
        blocks = []
        for i in range(depth):
            blocks += [
                nn.Conv2d(channels[i], channels[i+1], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels[i+1]),
                nn.ReLU(inplace=True),
            ]
            # Downsample every 2 blocks
            if (i % 2) == 1:
                blocks.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(width, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
        x = self.dropout(x)
        return self.head(x)


class ResNetBackbone(nn.Module):
    def __init__(self, arch: Literal['resnet18','resnet34','resnet50']='resnet18', num_classes: int = 2, in_ch: int = 1):
        super().__init__()
        if tvm is None:
            raise RuntimeError('torchvision is required for ResNet models')
        if arch == 'resnet18':
            net = tvm.resnet18(weights=None)
        elif arch == 'resnet34':
            net = tvm.resnet34(weights=None)
        elif arch == 'resnet50':
            net = tvm.resnet50(weights=None)
        else:
            raise ValueError(f'Unknown arch: {arch}')
        # Adapt first conv to 1-channel
        if in_ch != 3:
            w = net.conv1.weight
            net.conv1 = nn.Conv2d(in_ch, net.conv1.out_channels, kernel_size=net.conv1.kernel_size,
                                   stride=net.conv1.stride, padding=net.conv1.padding, bias=False)
            if w.shape[1] == 3 and in_ch == 1:
                net.conv1.weight.data = w.data.mean(dim=1, keepdim=True)
        # Replace fc
        feat_dim = net.fc.in_features
        net.fc = nn.Linear(feat_dim, num_classes)
        self.net = net

    def forward(self, x):
        return self.net(x)


def create_model(name: str, num_classes: int = 2, in_ch: int = 1):
    name = name.lower()
    if name == 'smallcnn':
        return SmallCNN(in_ch=in_ch, num_classes=num_classes)
    if name.startswith('deepcnn'):
        # deepcnn:w64:d8
        width = 64
        depth = 6
        if ':' in name:
            parts = name.split(':')
            for p in parts[1:]:
                if p.startswith('w'):
                    width = int(p[1:])
                if p.startswith('d'):
                    depth = int(p[1:])
        return DeepCNN(in_ch=in_ch, width=width, depth=depth, num_classes=num_classes)
    if name in ('resnet18','resnet34','resnet50'):
        return ResNetBackbone(arch=name, num_classes=num_classes, in_ch=in_ch)
    raise ValueError(f'Unknown model name: {name}')

