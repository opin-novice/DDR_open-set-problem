import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=1000, backbone_fc=True, pretrained=True):
        super(EfficientNetB0, self).__init__()
        
        # Load pre-trained EfficientNet-B0
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = models.efficientnet_b0(weights=weights)
        
        # Remove the classifier to get features
        self.features = self.model.features
        self.avgpool = self.model.avgpool
        
        self.out_channels = 1280  # EfficientNet-B0 output channels
        
        self.backbone_fc = backbone_fc
        if backbone_fc:
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(self.out_channels, num_classes),
            )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        if self.backbone_fc:
            x = self.classifier(x)
            
        return x

def efficientnet_b0(num_classes=1000, backbone_fc=True):
    """
    Constructs a EfficientNet-B0 model.
    """
    return EfficientNetB0(num_classes=num_classes, backbone_fc=backbone_fc, pretrained=True)
