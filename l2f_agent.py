import torch
import torch.nn as nn


class ConvertedModel(nn.Module):
    """Pretrained 146-64-64-4 controller (weights in l2f_agent.pth)."""
    def __init__(self):
        super().__init__()
        self.layer0 = nn.Linear(146, 64)
        self.layer1 = nn.Linear(64, 64)
        self.layer2 = nn.Linear(64, 4)

    def forward(self, x):
        x = torch.tanh(self.layer0(x))
        x = torch.tanh(self.layer1(x))
        return torch.tanh(self.layer2(x))
