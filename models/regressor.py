import torch
import torch.nn as nn


class Regressor(nn.Module):
    def __init__(self, input_dim=7):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.ln = nn.LayerNorm(32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
