import torch
from torch import nn
from torch.nn.utils import parameters_to_vector
from torch.functional import F
from itertools import chain


class SurrogateNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SurrogateNetwork, self).__init__()

        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, 25),
            nn.Tanh(),
            nn.Linear(25, 1),
            nn.Softplus()
        )

    def forward(self, x):
        return self.feed_forward(x) + 1

    def freeze_model(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_model(self):
        for param in self.parameters():
            param.requires_grad = True

    @property
    def get_parameter_vector(self) -> torch.Tensor:
        return parameters_to_vector(self.feed_forward.parameters())


class TreeNet(nn.Module):
    def __init__(self, input_dim):
        super(TreeNet, self).__init__()

        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 10),
            nn.Tanh(),
            nn.Linear(10, 1)
        )
        self.surrogate_network = SurrogateNetwork(self.get_parameter_vector.shape[0])
        self.surrogate_network.freeze_model()

    def forward(self, x):
        return self.feed_forward(x)

    def compute_APL_prediction(self):
        return self.surrogate_network(self.get_parameter_vector)

    def freeze_model(self):
        for param in self.feed_forward.parameters():
            param.requires_grad = False

    def unfreeze_model(self):
        for param in self.feed_forward.parameters():
            param.requires_grad = True

    def reset_outer_weights(self):
        """
        Reset all weights of the feed forward network for random restarts.
        Required for initial surrogate data.
        :return:
        """
        self.feed_forward.apply(lambda m: isinstance(m, nn.Linear) and m.reset_parameters())

    def reset_surrogate_weights(self):
        """
        Reset all weights of the feed forward network for random restarts.
        Required for initial surrogate data.
        :return:
        """
        self.surrogate_network.apply(lambda m: isinstance(m, nn.Linear) and m.reset_parameters())

    @property
    def get_parameter_vector(self) -> torch.Tensor:
        return parameters_to_vector(self.feed_forward.parameters())

# Custom Neural Network generator
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

    def get_parameter_vector(self):
        return torch.cat([torch.flatten(x) for x in self.parameters()])
