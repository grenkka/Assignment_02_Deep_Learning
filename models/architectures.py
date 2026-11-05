import torch
import torch.nn as nn
import timm

class SqueezeAndExcitation(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block (Hu et al., CVPR 2018).
    Explicitly models channel-wise interdependencies.
    - Squeeze: Global Average Pooling aggregates spatial information (H, W) -> (1, 1).
    - Excitation: 2 FC layers with reduction ratio r and sigmoid activation adaptively recalibrate channel weights.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced_channels = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        weights = self.fc(x).view(b, c, 1, 1)
        return x * weights


class ResNet50WithSE(nn.Module):
    """
    ResNet-50 modified with custom Squeeze-and-Excitation blocks
    inserted into the output of each residual stage.
    """
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()
        # Load backbone without default pooling and head
        self.backbone = timm.create_model('resnet50', pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features # 2048
        
        # Custom SE block inserted on final stage feature maps
        self.custom_se = SqueezeAndExcitation(in_features, reduction=16)
        
        # Pooling + Multi-label Classification Head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # Standard stem
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)
        
        # Residual layers
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        # Pass through custom building block
        x = self.custom_se(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        feat = self.global_pool(feat).flatten(1)
        feat = self.dropout(feat)
        return self.classifier(feat)


class BaselineResNet50(nn.Module):
    """
    Standard Pretrained ResNet-50 without custom SE block for ablation comparison.
    """
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model('resnet50', pretrained=pretrained, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


if __name__ == '__main__':
    dummy = torch.randn(2, 3, 224, 224)
    se_net = ResNet50WithSE(num_classes=5, pretrained=False)
    out = se_net(dummy)
    print("ResNet50WithSE output shape:", out.shape)
    assert out.shape == (2, 5), "Output shape mismatch!"
    print("Architecture verified successfully.")

