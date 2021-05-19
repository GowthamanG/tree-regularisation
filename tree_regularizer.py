import torch
from torch import nn
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Optimizer
from torch.functional import F
import numpy as np
import copy
from sklearn.tree import DecisionTreeClassifier
from typing import Callable, Optional
from torch.autograd import Variable
from sklearn.model_selection import train_test_split


class TreeRegularizedLoss(nn.Module):

    def __init__(self, loss_function, strength):
        super().__init__()
        self.strength = strength
        self.loss = loss_function

    def forward(self, predicted, labels, predicted_tree_length):
        loss = self.loss(predicted, labels)
        regularization = self.strength * predicted_tree_length
        return loss + regularization

class Tree_Regularizer(_Loss):

    def __init__(self, strength, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(Tree_Regularizer, self).__init__(size_average, reduce, reduction)
        self.strength = strength

    def forward(self, input, target, parameters, regularization_term):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        return F.binary_cross_entropy_with_logits(input=input, target=target) + self.strength * regularization_term


def compute_regularization_term(input, surrogate_model):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #input = input.flatten()
    #zeros = torch.zeros(surrogate_model.fc1.in_features - input.size()[0]).to(device) # additional zeros since the input could be smaller
    #input = torch.cat((input, zeros))
    surrogate_model.to(device)
    surrogate_model.eval()
    #regularization_term = surrogate_model(input.reshape(1, -1))
    regularization_term = surrogate_model(input)

    return regularization_term


def params_to_1D_vector(model_parameters):
    parameters = []
    for param in model_parameters:
        parameters.append(torch.flatten(param))

    return torch.cat(parameters, dim=0)

def train_surrogate_model(params, APLs, epsilon, learning_rate=10e-3, retrain=False, current_surrogate_model=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #APLs = APLs - torch.mean(APLs)

    if retrain:
        model = SurrogateNetwork(params.size()[1])
    else:
        if current_surrogate_model is not None:
            model = current_surrogate_model
        else:
            model = SurrogateNetwork(params.size()[1])

    X_train, X_test, y_train, y_test = train_test_split(params.cpu().detach().numpy(), APLs.numpy(), test_size=0.10, random_state=42)

    model.to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    model.train()
    num_epochs = 250
    batch_size = 100

    #train_data = TensorDataset(copy.deepcopy(Variable(params, requires_grad=False)), APLs)
    #train_loader = DataLoader(dataset=train_data, batch_size=batch_size)

    X_train = torch.tensor(X_train, dtype=torch.float)
    y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float)
    X_test = torch.tensor(X_test, dtype=torch.float)
    y_test = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float)

    data_train = TensorDataset(X_train, y_train)
    data_train_loader = DataLoader(dataset=data_train, batch_size=batch_size)
    data_test = TensorDataset(X_test, y_test)
    data_test_loader = DataLoader(dataset=data_test, batch_size=batch_size)

    training_loss = []

    for epoch in range(num_epochs):
        running_loss = []

        for i, batch in enumerate(data_train_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()
            y_hat = model(x_batch)

            model_parameters = params_to_1D_vector(model.parameters())

            loss = criterion(input=y_hat, target=y_batch) + epsilon * torch.norm(model_parameters, 2)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item()/(np.var(APLs.detach().cpu().numpy())+0.01))
        print(f'Surrogate training, epoch: {epoch + 1}/{num_epochs}, loss: {np.array(running_loss).mean():.4f}')
        training_loss.append(np.array(running_loss).mean())

    model.eval()
    y_predicted = []
    loss_with_test_data = []
    with torch.no_grad():
        # Test with training data
        for i, batch in enumerate(data_test_loader):
            x, y = batch[0].to(device), batch[1].to(device)
            y_hat = model(x)
            y_predicted.append(y_hat.cpu().detach().numpy())

            model_parameters = params_to_1D_vector(model.parameters())

            loss = criterion(input=y_hat, target=y)

            loss_with_test_data.append(loss.item())

        y_predicted = np.vstack(y_predicted)

    print(f'Loss: {np.mean(loss_with_test_data)}')

    for i in range(len(y_predicted)):
        print(f'y: {data_test_loader.dataset[i][1]}, y_hat : {y_predicted[i]}')


    #print('Dim input:', weights.shape)
    #print(weights)
    return model, training_loss


class SurrogateNetwork(nn.Module):
    def __init__(self, d_in):
        super(SurrogateNetwork, self).__init__()
        self.fc1 = nn.Linear(d_in, 25)
        self.ReLu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.fc2 = nn.Linear(25, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ReLu(x)
        x = self.fc2(x)
        #x = self.softplus(x)
        return x

# todo: Perhaps not used anymore --> delete
class SumOfSquareLossSurrogate(_Loss):
    def __init__(self, strength) -> None:
        super(SumOfSquareLossSurrogate, self).__init__()
        self.strength = strength

    def forward(self, input: Tensor, target: Tensor, model_parameters):
        # todo: implement correctly
        loss = torch.sum(torch.pow(target - input, 2)) + self.strength * torch.pow(torch.linalg.norm(model_parameters, 2), 2)
        return loss

