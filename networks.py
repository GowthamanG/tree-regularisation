import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.functional import F
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from itertools import chain
from typing import Iterator
from torch.nn.parameter import Parameter

np.random.seed(5555)
torch.random.manual_seed(5255)

import warnings
warnings.filterwarnings('ignore')

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class SurrogateNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SurrogateNetwork, self).__init__()

        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, 25),
            nn.ReLU(),
            nn.Linear(25, 1),
            nn.Softplus()
        )

    def forward(self, x):
        return self.feed_forward(x)

    def freeze_model(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_model(self):
        for param in self.parameters():
            param.requires_grad = True

    def parameters_to_vector(self) -> torch.Tensor:
        return parameters_to_vector(self.feed_forward.parameters())

    def vector_to_parameters(self, parameter_vector):
        vector_to_parameters(parameter_vector, self.feed_forward.parameters())


class TreeNet(nn.Module):
    def __init__(self, input_dim):
        super(TreeNet, self).__init__()

        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        self.surrogate_network = SurrogateNetwork(self.parameters_to_vector().numel())
        self.surrogate_network.freeze_model()

        self.random_seeds = np.random.randint(1, 100, 10)

    def forward(self, x):
        return self.feed_forward(x)

    def compute_APL(self, X, ccp_alpha):

        def sequence_to_samples(tensor):
            sequence_array = [tensor[idx, :, :] for idx in range(tensor.shape[0])]
            return np.vstack(sequence_array)

        self.freeze_model()
        self.eval()
        y_tree = self(X).cpu().detach().numpy()
        self.unfreeze_model()
        self.train()
        # y_tree = sequence_to_samples(y_tree)
        # y_tree = np.argmax(y_tree, axis=1)

        X_tree = X.cpu().detach().numpy()
        # X_tree = sequence_to_samples(X_tree)

        path_lengths = []

        for random_state in self.random_seeds:
            tree = DecisionTreeClassifier(min_samples_leaf=3, random_state=random_state)
            y_tree = np.where(y_tree > 0.5, 1, 0)
            tree.fit(X_tree, y_tree)

            average_path_length = np.mean(np.sum(tree.tree_.decision_path(X_tree), axis=1))
            maximum_path_length = tree.get_depth()
            path_lengths.append(average_path_length)

            del tree

        return np.mean(path_lengths)

    def compute_APL_prediction(self):
        """
        Computes the average-path-length (APL) prediction with the surrogate model using the
        current target model parameters W as input.

        :return: APL prediction as the regulariser Omega(W)
        """
        # x_transformed = self.scaler.transform(self.parameters_to_vector().cpu().detach().numpy().reshape(1, -1))
        # x_transformed = torch.from_numpy(x_transformed).to('cuda:0')
        # return self.surrogate_network(x_transformed)

        return self.surrogate_network(self.parameters_to_vector())

    def freeze_model(self):
        for param in self.feed_forward.parameters():
            param.requires_grad = False

    def unfreeze_model(self):
        for param in self.feed_forward.parameters():
            param.requires_grad = True

    def freeze_bias(self):
        for name, param in self.feed_forward.named_parameters():
            if 'bias' in name:
                param.requires_grad = False

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

    def parameters_to_vector(self) -> torch.Tensor:
        return parameters_to_vector(self.feed_forward.parameters())

    def vector_to_parameters(self, parameter_vector):
        vector_to_parameters(parameter_vector, self.feed_forward.parameters())


class GRU_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1):
        super(GRU_RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]

        return self.fc(out)


class TreeGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1):
        super(TreeGRU, self).__init__()

        #self.input_size = input_size
        #self.hidden_size = hidden_size
        #self.num_layers = num_layers
        #self.num_classes = num_classes

        # self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        # self.fc = nn.Linear(self.hidden_size, self.num_classes)
        # self.softmax = nn.Softmax(dim=2)

        self.gru_rnn = GRU_RNN(input_size, hidden_size, num_classes, num_layers)

        self.surrogate_network = SurrogateNetwork(self.parameters_to_vector().numel())
        self.surrogate_network.freeze_model()

        self.random_seeds = np.random.randint(1, 100, 10)

    def forward(self, x):
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # out, _ = self.gru(x, h0)
        # out = out[:, -1, :]
        # out = self.fc(out)
        #
        # return out
        return self.gru_rnn(x)

    def compute_APL(self, X, ccp_alpha):

        def sequence_to_samples(tensor):
            sequence_array = [tensor[idx, :, :] for idx in range(tensor.shape[0])]
            return np.vstack(sequence_array)

        self.freeze_model()
        self.eval()
        y_tree = self(X).cpu().detach().numpy()
        self.unfreeze_model()
        self.train()
        # y_tree = sequence_to_samples(y_tree)
        # y_tree = np.argmax(y_tree, axis=1)

        X_tree = X[:, -1, :].cpu().detach().numpy()
        # X_tree = sequence_to_samples(X_tree)

        path_lengths = []

        """What is the correct way to create a pruned tree?
        If min_samples_leaf would be a float, this would reflect also the total numbers of samples.
        Otherwise, the trees could get more complex with bigger datasets."""
        for random_state in self.random_seeds:
            tree = DecisionTreeClassifier(min_samples_leaf=5, random_state=random_state)
            y_tree = np.where(y_tree > 0.5, 1, 0)
            tree.fit(X_tree, y_tree)

            average_path_length = np.mean(np.sum(tree.tree_.decision_path(X_tree), axis=1))
            maximum_path_length = tree.get_depth()
            path_lengths.append(average_path_length)

            del tree

        return np.mean(path_lengths)

    def compute_APL_prediction(self):
        """
        Computes the average-path-length (APL) prediction with the surrogate model using the
        current target model parameters W as input.

        :return: APL prediction as the regulariser Omega(W)
        """

        return self.surrogate_network(self.parameters_to_vector())

    def freeze_model(self):
        for param in self.gru_rnn.parameters():
            param.requires_grad = False

    def unfreeze_model(self):
        for param in self.gru_rnn.parameters():
            param.requires_grad = True

    def freeze_bias(self):
        for name, param in self.gru_rnn.named_parameters():
            if 'bias' in name:
                param.requires_grad = False

    def reset_outer_weights(self):
        """
        Reset all weights of the feed forward network for random restarts.
        Required for initial surrogate data.
        :return:
        """
        self.gru_rnn.apply(lambda m: isinstance(m, nn.Linear) and m.reset_parameters())

    def reset_surrogate_weights(self):
        """
        Reset all weights of the feed forward network for random restarts.
        Required for initial surrogate data.
        :return:
        """
        self.surrogate_network.apply(lambda m: isinstance(m, nn.Linear) and m.reset_parameters())

    def parameters_to_vector(self) -> torch.Tensor:
        return parameters_to_vector(self.gru_rnn.parameters())

    def vector_to_parameters(self, parameter_vector):
        vector_to_parameters(parameter_vector, self.gru_rnn.parameters())

