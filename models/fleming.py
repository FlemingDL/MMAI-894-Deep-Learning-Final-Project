"""Team fleming's custom CNN

Load the model by specifying it in the params.json file in experiments

"""

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


class FlemingModel_v1(nn.Module):

    def __init__(self, num_classes=2):
        super(FlemingModel_v1, self).__init__()
        self.features1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.features2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6, 6))
        self.classifier1 = nn.Sequential(
            nn.Linear(in_features=12 * 6 * 6, out_features=120),
            nn.ReLU())
        self.classifier2 = nn.Sequential(
            nn.Linear(in_features=120, out_features=60),
            nn.ReLU())
        self.classifier3 = nn.Sequential(
            nn.Linear(60, num_classes)
        )

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier1(x)
        x = self.classifier2(x)
        x = self.classifier3(x)
        return x


class FlemingModel_v2(nn.Module):

    def __init__(self, num_classes=2):
        super(FlemingModel_v2, self).__init__()
        self.features1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11),
            nn.ReLU()
        )
        self.features2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5),
            nn.ReLU()
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6, 6))
        self.classifier1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=128 * 6 * 6, out_features=392),
            nn.ReLU())
        self.classifier2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=392, out_features=392),
            nn.ReLU())
        self.classifier3 = nn.Sequential(
            nn.Linear(392, num_classes)
        )

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier1(x)
        x = self.classifier2(x)
        x = self.classifier3(x)
        return x


# class FlemingModel_v3(nn.Module):
#
#     def __init__(self, num_classes=2):
#         super(FlemingModel_v3, self).__init__()
#         self.features1 = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2)
#         )
#         self.features2 = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#         )
#         self.features3 = nn.Sequential(
#             nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=32),
#         )
#         self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6, 6))
#         self.classifier1 = nn.Sequential(
#             nn.Linear(in_features=128 * 6 * 6, out_features=392),
#             nn.ReLU())
#         self.classifier2 = nn.Sequential(
#             nn.Linear(in_features=392, out_features=392),
#             nn.ReLU())
#         self.classifier3 = nn.Sequential(
#             nn.Linear(392, num_classes)
#         )
#
#     def forward(self, x):
#         x = self.features1(x)
#         x = self.features2(x)
#         x = self.features3(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier1(x)
#         x = self.classifier2(x)
#         x = self.classifier3(x)
#         return x

class FlemingModel_v3(nn.Module):

    def __init__(self, num_classes=2):
        super(FlemingModel_v3, self).__init__()
        self.features1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.features2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.features3 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=32),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6, 6))
        self.classifier1 = nn.Sequential(
            nn.Linear(in_features=128 * 6 * 6, out_features=392),
            nn.ReLU())
        self.classifier2 = nn.Sequential(
            nn.Linear(in_features=392, out_features=392),
            nn.ReLU())
        self.classifier3 = nn.Sequential(
            nn.Linear(392, num_classes)
        )

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.features3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier1(x)
        x = self.classifier2(x)
        x = self.classifier3(x)
        return x