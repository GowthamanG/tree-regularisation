import warnings
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from sklearn.tree import DecisionTreeClassifier
import numpy as np


np.random.seed(5555)
torch.random.manual_seed(5255)

warnings.filterwarnings('ignore')

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class SurrogateNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SurrogateNetwork, self).__init__()

        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
            nn.Softplus()
        )

    def forward(self, x):
        return self.feed_forward(x)

    def freeze_model(self):
        """
        Disable model updates by gradient-descent by freezing the model parameters.
        """
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_model(self):
        """
        Enable model updates by gradient-descent by unfreezing the model parameters.
        """
        for param in self.parameters():
            param.requires_grad = True

    def parameters_to_vector(self) -> torch.Tensor:
        """
        Convert model parameters to vector.
        """
        return parameters_to_vector(self.feed_forward.parameters())

    def vector_to_parameters(self, parameter_vector):
        """
        Overwrite the model parameters with given parameter vector.
        """
        vector_to_parameters(parameter_vector, self.feed_forward.parameters())


class TreeNet(nn.Module):
    def __init__(self, input_dim, min_samples_leaf=1):
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
        self.surrogate_network = SurrogateNetwork(self.parameters_to_vector().numel())
        self.surrogate_network.freeze_model()

        self.min_samples_leaf = min_samples_leaf

        self.random_seeds = np.random.randint(1, 100, 10)

    def forward(self, x):
        return self.feed_forward(x)

    def compute_APL(self, X):
        """
        Compute average decision path length given input data. It computes the how many decision nodes one has to
        traverse on average for one data instance.

        Parameters
        -------

        X: Input features

        Returns
        -------

        average decision path lengths, taking the average from several runs with different random seeds

        """

        def sequence_to_samples(tensor):
            sequence_array = [tensor[idx, :, :] for idx in range(tensor.shape[0])]
            return np.vstack(sequence_array)

        self.freeze_model()
        self.eval()
        y_tree = self(X)
        y_tree = torch.where(y_tree > 0.5, 1, 0).detach().cpu().numpy()
        self.unfreeze_model()
        self.train()

        X_tree = X.cpu().detach().numpy()

        path_lengths = []

        for random_state in self.random_seeds:
            tree = DecisionTreeClassifier(min_samples_leaf=self.min_samples_leaf, random_state=random_state)
            tree.fit(X_tree, y_tree)
            average_path_length = np.mean(np.sum(tree.tree_.decision_path(X_tree), axis=1))
            path_lengths.append(average_path_length)

            del tree

        return np.mean(path_lengths)

    def compute_APL_prediction(self):
        """
        Computes the average-path-length (APL) prediction with the surrogate model using the
        current target model parameters W as input.

        Returns
        -------

        APL prediction as the regulariser Omega(W)
        """
        return self.surrogate_network(self.parameters_to_vector())

    def freeze_model(self):
        """
        Disable model updates by gradient-descent by freezing the model parameters.
        """

        for param in self.feed_forward.parameters():
            param.requires_grad = False

    def unfreeze_model(self):
        """
        Enable model updates by gradient-descent by unfreezing the model parameters.
        """
        for param in self.feed_forward.parameters():
            param.requires_grad = True

    def freeze_bias(self):
        """
        Disable model updates by gradient-descent by freezing the biases.
        """
        for name, param in self.feed_forward.named_parameters():
            if 'bias' in name:
                param.requires_grad = False

    def reset_outer_weights(self):
        """
        Reset all weights of the feed forward network for random restarts.
        Required for initial surrogate data.
        """
        self.feed_forward.apply(lambda m: isinstance(m, nn.Linear) and m.reset_parameters())

    def reset_surrogate_weights(self):
        """
        Reset all weights of the feed forward network for random restarts.
        Required for initial surrogate data.
        """
        self.surrogate_network.apply(lambda m: isinstance(m, nn.Linear) and m.reset_parameters())

    def parameters_to_vector(self) -> torch.Tensor:
        """
        Convert model parameters to vector.
        """

        return parameters_to_vector(self.feed_forward.parameters())

    def vector_to_parameters(self, parameter_vector):
        """
        Overwrite the model parameters with given parameter vector.
        """
        vector_to_parameters(parameter_vector, self.feed_forward.parameters())
