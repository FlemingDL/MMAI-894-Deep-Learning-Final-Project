import torch
import torch.nn as nn


class FlemingModel(nn.Module):

    def __init__(self, num_classes=2):
        super(FlemingModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6, 6))
        self.classifier = nn.Sequential(
            nn.Linear(in_features=12 * 6 * 6, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=60),
            nn.ReLU(),
            nn.Linear(60, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
