import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_features: int = 64, hidden: int = 64, classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, classes),
        )

    def forward(self, x):
        return self.net(x)
