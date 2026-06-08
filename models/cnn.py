try:
    import torch
    from torch import nn
except ImportError:  # Keeps lightweight EDA environments importable.
    torch = None
    nn = None


class SimpleCNN(nn.Module if nn else object):
    def __init__(self, num_classes=6):
        if nn is None:
            raise ImportError("Install PyTorch to use SimpleCNN.")
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.flatten(1))
