import torch
from torch import nn
from torch.functional import F


class Net1(nn.Module):

    def __init__(self, input_dim):
        super(Net1, self).__init__()
        self.input = nn.Linear(input_dim, 50)
        self.output = nn.Linear(50, 1)

    def forward(self, x):
        fc1 = torch.tanh(self.input(x))
        y_hat = self.output(fc1)

        return y_hat

    def parameters_to_vector(self):
        parameters = []
        for param in self.parameters():
            parameters.append(torch.flatten(param))

        return torch.cat(parameters, dim=0)


class Net2(nn.Module):  # from paper

    def __init__(self, input_dim):
        super(Net2, self).__init__()
        self.input = nn.Linear(input_dim, 100)
        self.hidden_1 = nn.Linear(100, 100)
        self.hidden_2 = nn.Linear(100, 10)
        self.output = nn.Linear(10, 1)

        self.ReLu = nn.ReLU()

    def forward(self, x):
        fc1 = F.relu(self.input(x))
        fc2 = F.relu(self.hidden_1(fc1))
        fc3 = F.relu(self.hidden_2(fc2))

        y_hat = self.output(fc3)

        return y_hat

    def parameters_to_vector(self):
        parameters = []
        for param in self.parameters():
            parameters.append(torch.flatten(param))

        return torch.cat(parameters, dim=0)


class Net3(nn.Module):

    def __init__(self, input_dim):
        super(Net3, self).__init__()
        self.input = nn.Linear(input_dim, 64)
        self.hidden_1 = nn.Linear(64, 32)
        self.hidden_2 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)

    def forward(self, x):
        fc1 = F.relu(self.input(x))
        fc2 = F.relu(self.hidden_1(fc1))
        fc3 = F.relu(self.hidden_2(fc2))

        y_hat = self.output(fc3)

        return y_hat

    def parameters_to_vector(self):
        parameters = []
        for param in self.parameters():
            parameters.append(torch.flatten(param))

        return torch.cat(parameters, dim=0)


class MyNet(nn.Module):

    def __init__(self, dimensions: list):
        super(MyNet, self).__init__()
        self.layers = []

        for i in range(len(dimensions) - 1):
            self.layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))

    def forward(self, x):
        output = F.relu(self.layers[0](x))
        for i in range(1, len(self.layers)):
            output = F.relu(self.layers[i](output))

        return output

    def parameters_to_vector(self):
        parameters = []
        for param in self.parameters():
            parameters.append(torch.flatten(param))

        return torch.cat(parameters, dim=0)
