import torch
import torch.nn as nn


class TestModel(nn.Module):
    def __init__(self, in_features=12):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(),

        )
