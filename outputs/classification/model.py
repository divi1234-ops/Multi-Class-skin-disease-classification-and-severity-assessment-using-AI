import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)
class SkinDiseaseClassifier(nn.Module):

    def __init__(self, num_classes=7):

        super().__init__()

        self.model = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT
        )

        in_features = self.model.classifier[1].in_features

        self.model.classifier = nn.Sequential(

            nn.Dropout(0.3),

            nn.Linear(
                in_features,
                num_classes
            )

        )

    def forward(self, x):

        return self.model(x)