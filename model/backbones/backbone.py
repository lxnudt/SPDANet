import torch
import torch.nn as nn
from torch import Tensor
from typing import Type, Callable, Union, List, Optional

class BasicBlock(nn.Module):
    """基础残差块，适用于ResNet18/34"""
    expansion: int = 1  # 输出通道扩展系数

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    """Bottleneck残差块，适用于ResNet50/101/152"""
    expansion: int = 4  # 输出通道扩展系数
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    """灵活的ResNet实现"""
    
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        in_channels: int = 3
    ) -> None:
        super().__init__()
        self.in_channels = 64
        
        # 初始卷积层
        #self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #self.bn1 = nn.BatchNorm2d(64)
        #self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 残差层构建
        self.layer1 = self._make_layer(block, 64, layers[1-1], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[2-1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[3-1], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[4-1], stride=2)
        
        # 分类器
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels,
                          out_channels * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels * block.expansion))

        layers = []
        layers.append(block(
            self.in_channels,
            out_channels,
            stride,
            downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(
                self.in_channels,
                out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = []
        #x = self.conv1(x)
        #x = self.bn1(x)
        #x = self.relu(x)
        #x = self.maxpool(x)

        x_conv1 = self.layer1(x)
        x_conv2 = self.layer2(x_conv1)
        out.append(x_conv2)
        x_conv3 = self.layer3(x_conv2)
        out.append(x_conv3)
        x_conv4 = self.layer4(x_conv3)
        out.append(x_conv4)

        #x = self.avgpool(x)
        #x = torch.flatten(x, 1)
        #x = self.fc(x)
        return out

def build_backbone(model_type: str, num_classes: int = 1000, in_channels: int = 3) -> ResNet:
    """构建指定类型的ResNet模型
    Args:
        model_type (str): 模型类型 ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
        num_classes (int): 分类类别数
        in_channels (int): 输入通道数
        
    Returns:
        ResNet: 构建的模型实例
    """
    config = {
        'resnet18': (BasicBlock, [2, 2, 2, 2]),
        'resnet34': (BasicBlock, [3, 4, 6, 3]),
        'resnet50': (Bottleneck, [3, 4, 6, 3]),
        'resnet101': (Bottleneck, [3, 4, 23, 3]),
        'resnet152': (Bottleneck, [3, 8, 36, 3])
    }
    
    if model_type not in config:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: {list(config.keys())}")
    
    block_type, layers = config[model_type]
    return ResNet(block_type, layers, num_classes, in_channels)

# 使用示例
if __name__ == '__main__':
    # 构建不同版本的ResNet
    resnet18 = build_backbone('resnet18', num_classes=1000)
    resnet34 = build_backbone('resnet34', num_classes=1000)
    resnet101 = build_backbone('resnet101', num_classes=1000)
    
    # 打印模型结构
    print(resnet18)
    print(f"\nResNet18参数数量: {sum(p.numel() for p in resnet18.parameters()) / 1e6:.2f}M")
    
    # 测试前向传播
    x = torch.randn(2, 3, 224, 224)  # 2张224x224 RGB图像
    output = resnet18(x)
    print(f"\n输出形状: {output.shape}")  # 应为 torch.Size([2, 1000])

