import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import vgg19, VGG19_Weights

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # ✅ FIX: Use weights instead of 'pretrained=True'
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:36]  
        self.model = nn.Sequential(*vgg)

    def forward(self, x):
        return self.model(x)
