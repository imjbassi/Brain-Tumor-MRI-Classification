# model.py
import torch
import torch.nn as nn
import torchvision.models as models

class TumorClassifier(nn.Module):
    """
    A PyTorch model for brain tumor classification using a pretrained ResNet-18.
    """
    def __init__(self, num_classes=4):
        super(TumorClassifier, self).__init__()
        # Load pretrained ResNet18
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Replace the final fully-connected layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)
