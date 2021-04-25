import torch
from torch import nn
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Optimizer
from torch.functional import F
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from typing import Callable, Optional


class Tree_Regularizer(_Loss):

    def __init__(self, strength) -> None:
        super(Tree_Regularizer, self).__init__()
        self.strength = strength

    # Todo: implement tree regularizer
    def forward(self, input, target, weights, model):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        regularization_term = model(weights)
        return F.mse_loss(input, target) + self.strength * regularization_term


def train_surrogate_model(weights, APLs, strength, learning_rate=0.001, retrain=False, current_surrogate_model=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if retrain:
        model = SurrogateModel(weights.size()[1])
    else:
        if current_surrogate_model is not None:
            model = current_surrogate_model
        else:
            model = SurrogateModel(weights.size()[1])
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    model.train()
    num_epochs = 250
    batch_size = 100

    train_data = TensorDataset(weights, APLs)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size)

    training_loss = []

    for epoch in range(num_epochs):
        running_loss = []

        for i, batch in enumerate(train_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            y_hat = model(x_batch)

            parameters = []
            for param in model.parameters():
                parameters.append(torch.flatten(param.data))
            weights = torch.cat(parameters)

            loss = criterion(input=y_batch, target=y_hat)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())
        print(f'Surrogate training, epoch: {epoch + 1}/{num_epochs}, loss: {np.array(running_loss).mean():.4f}')
        training_loss.append(np.array(running_loss).mean())

    return model, training_loss


class SurrogateModel(nn.Module):
    def __init__(self, d_in):
        super(SurrogateModel, self).__init__()
        self.fc1 = nn.Linear(d_in, 25)
        self.ReLu = nn.ReLU()
        self.fc2 = nn.Linear(25, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ReLu(x)
        x = self.fc2(x)
        return x


class SumOfSquareLossSurroage(_Loss):
    def __init__(self, strength) -> None:
        super(SumOfSquareLossSurroage, self).__init__()
        self.strength = strength

    def forward(self, input: Tensor, target: Tensor, model_parameters):
        # todo: implement correctly
        loss = torch.sum(torch.pow(target - input, 2)) + self.strength * torch.pow(torch.linalg.norm(model_parameters, 2), 2)
        #loss = nn
        return loss

