import torch
import torch.nn as nn
import numpy as np

from config import device
from utils.assistive_functions import to_tensor


class MLP(nn.Module):
    def __init__(self, dim_in = 12,dim_out = 1):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(dim_in, 15),
            nn.Sigmoid(),
            nn.Linear(15, 20),
            nn.Sigmoid(),
            nn.Linear(20, 14),
            nn.Sigmoid(),
            nn.Linear(14, dim_out),
        )


    def forward(self, x):
        return  self.mlp(x)
