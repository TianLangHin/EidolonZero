"""
    Summary of convolutional net architecture:
        - Take in 8x8x18 input tensor, extract features and return the value and policy.
        - Value is a single number between -1 and +1 where -1 is a loss and 1 is a win.
          0 is supposed to be a draw.
        - Policy is an 8x8x73 (flattened as a 1x4672) tensor. It is left in this flattened form because the loss function
          will conduct back propagation easier.

    Can test different numbers of filters and more/less layers, just trying to be mindful of training times.
"""

import torch

class ConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.res_tower = torch.nn.Sequential(
            torch.nn.Conv2d(18, 256, kernel_size=3, stride=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
        )
        self.value_head = torch.nn.Sequential(
            torch.nn.Conv2d(256, 1, kernel_size=1, stride=1),
            torch.nn.BatchNorm2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(4 * 4 * 1, 1),
            torch.nn.Tanh(),
        )
        self.policy_head = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(5 * 5 * 256, 5000),
            torch.nn.ReLU(),
            torch.nn.Linear(5000, 4672),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        res_tower_output = self.res_tower(x)
        value = self.value_head(res_tower_output).flatten()
        policy_raw = self.policy_head(res_tower_output)
        return value, policy_raw
