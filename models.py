import torch
from torch import nn
from torch.functional import F
import numpy as np
from sklearn.tree import DecisionTreeClassifier


class MLP(nn.Module):

    def __init__(self, d_in, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(d_in, hidden_size)
        self.ReLu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ReLu(x)
        x = self.fc2(x)
        return x
