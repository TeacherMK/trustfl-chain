from __future__ import annotations

import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        side = 7 if in_channels == 1 else 8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * side * side, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(dataset: str):
    if dataset == "fashion_mnist":
        return SmallCNN(1, 10)
    if dataset == "cifar10":
        return SmallCNN(3, 10)
    raise ValueError(f"Unsupported dataset: {dataset}")
