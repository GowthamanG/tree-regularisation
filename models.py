import torch
from torch import nn
from torch.functional import F
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class Net1(nn.Module):

    def __init__(self, d_in):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(d_in, 500)
        self.output = nn.Linear(500, 1)
        self.ReLu = nn.ReLU()


    def forward(self, x):
        x = self.fc1(x)
        x = self.ReLu(x)
        x = self.output(x)
        return x


class Net2(nn.Module):

    def __init__(self, d_in):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(d_in, 500)
        self.fc2 = nn.Linear(500, 1000)
        self.fc3 = nn.Linear(1000, 800)
        self.fc4 = nn.Linear(800, 300)
        self.output = nn.Linear(300, 1)

        self.ReLu = nn.ReLU()


    def forward(self, x):
        x = self.fc1(x)
        x = self.ReLu(x)
        x = self.fc2(x)
        x = self.ReLu(x)
        x = self.fc3(x)
        x = self.ReLu(x)
        x = self.fc4(x)
        x = self.ReLu(x)
        x = self.output(x)
        return x
