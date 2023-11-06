import timm
import torch.nn as nn


class ViTModel(nn.Module):
    def __init__(self, num_classes=100, pretrained=True):
        super(ViTModel, self).__init__()
        # Load a pre-trained Vision Transformer model
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained)

        # Replace the classifier head with the number of classes for CIFAR-100
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

    def forward(self, x):
        return self.vit(x)
