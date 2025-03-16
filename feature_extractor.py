import torch.nn as nn
import torchvision.models as models

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        self.feature_layers = nn.Sequential(*list(vgg.children())[:18])  # Fix: Close Sequential first
        self.feature_layers.eval()  # Then call eval()
        for param in self.feature_layers.parameters():
            param.requires_grad = False  # Freeze parameters

    def forward(self, x):
        return self.feature_layers(x)