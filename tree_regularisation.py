import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.functional import F
import numpy as np
from sklearn.model_selection import train_test_split


class SurrogateNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SurrogateNetwork, self).__init__()
        self.input = nn.Linear(input_dim, 25)
        self.output = nn.Linear(25, 1)

    def forward(self, x):
        fc1 = F.relu(self.input(x))
        y_hat = F.softplus(self.output(fc1))

        return y_hat

    def parameters_to_vector(self):
        parameters = []
        for param in self.parameters():
            parameters.append(torch.flatten(param))

        return torch.cat(parameters, dim=0)


class MySurrogateNetwork(nn.Module):
    def __init__(self, dimensions: list):
        super(MySurrogateNetwork, self).__init__()
        self.layers = []

        for i in range(len(dimensions) - 1):
            self.layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))

    def forward(self, x):
        output = F.relu(self.layers[0](x))
        for i in range(1, len(self.layers)):
            output = F.relu(self.layers[i](output))

        return F.softplus(output)

    def parameters_to_vector(self):
        parameters = []
        for param in self.parameters():
            parameters.append(torch.flatten(param))

        return torch.cat(parameters, dim=0)


def train_surrogate_model(params, APLs, epsilon, learning_rate=1e-2, retrain=False, current_optimizer=None,
                          current_surrogate_model=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if retrain:
        model = SurrogateNetwork(input_dim=params.size()[1])
    else:
        if current_surrogate_model is not None:
            model = current_surrogate_model
        else:
            model = SurrogateNetwork(input_dim=params.size()[1])

    X_train, X_test, y_train, y_test = train_test_split(params.cpu().detach().numpy(), APLs.numpy(), test_size=0.01,
                                                        random_state=42)

    model.to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if current_optimizer:
        optimizer.load_state_dict(current_optimizer)

    num_epochs = 100
    batch_size = 64

    X_train = torch.tensor(X_train, dtype=torch.float)
    y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float)
    X_test = torch.tensor(X_test, dtype=torch.float)
    y_test = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float)

    data_train = TensorDataset(X_train, y_train)
    data_train_loader = DataLoader(dataset=data_train, batch_size=batch_size)
    data_test = TensorDataset(X_test, y_test)
    data_test_loader = DataLoader(dataset=data_test, batch_size=batch_size)

    training_loss = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = []

        for i, batch in enumerate(data_train_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()
            y_hat = model(x_batch)
            loss = criterion(input=y_hat, target=y_batch) + epsilon * torch.norm(model.parameters_to_vector(), 2)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item() / (np.var(y_train.detach().cpu().numpy()) + 0.01))

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
            loss = criterion(input=y_hat, target=y)
            loss_with_test_data.append(loss.item())

        y_predicted = np.vstack(y_predicted)

    print(f'Testset loss: {np.mean(loss_with_test_data)}')

    for i in range(len(y_predicted)):
        print(f'y: {data_test_loader.dataset[i][1]}, y_hat : {y_predicted[i]}')

    return model, optimizer.state_dict(), training_loss
