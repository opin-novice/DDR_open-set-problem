import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, Bottleneck, ResNet50_Weights

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BottleneckCBAM(Bottleneck):
    """
    Bottleneck Block with CBAM (Channel + Spatial Attention)
    """
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BottleneckCBAM, self).__init__(inplanes, planes, stride, downsample, groups,
                                             base_width, dilation, norm_layer)
        
        # CBAM Modules
        self.ca = ChannelAttention(planes * 4)
        self.sa = SpatialAttention()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Apply CBAM
        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def resnet50_cbam(pretrained=True, progress=True, **kwargs):
    """
    Constructs a ResNet-50 model with CBAM.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet (weights loaded selectively)
    """
    # Create the model structure with CBAM blocks
    model = ResNet(BottleneckCBAM, [3, 4, 6, 3], **kwargs)
    
    if pretrained:
        print("Loading ResNet50 ImageNet weights for CBAM model...")
        # Load standard ResNet50 weights
        weights = ResNet50_Weights.IMAGENET1K_V1.get_state_dict(progress=progress)
        
        # Filter out weights that don't match (though BottleneckCBAM has same conv names, 
        # so most should match. The new attention layers will be initialized randomly)
        model_dict = model.state_dict()
        
        # Filter keys
        pretrained_dict = {k: v for k, v in weights.items() if k in model_dict and v.shape == model_dict[k].shape}
        
        # Update
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers from ImageNet.")
        
    return model
