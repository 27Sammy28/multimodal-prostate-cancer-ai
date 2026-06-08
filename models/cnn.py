import timm
import torch.nn as nn


class EfficientNetBaseline(nn.Module):

    def __init__(
        self,
        num_classes=6
    ):
        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=0
        )

        self.classifier = nn.Linear(
            self.backbone.num_features,
            num_classes
        )

    def forward(self, x):

        features = self.backbone(x)

        return self.classifier(features)
