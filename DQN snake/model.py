import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, channels, w, num_classes=4):
        super(DQN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # nn.Linear(16, 16),
            # nn.BatchNorm2d(16),
            # nn.ReLU(inplace=True),
        )

        self.classifier = nn.Linear(576, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # Dropout?
        x = self.classifier(x)
        return x
