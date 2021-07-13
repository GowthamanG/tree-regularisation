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
        y_hat = F.softplus(self.output(fc1)) + 1

        return y_hat

    def parameters_to_vector(self):
        parameters = []
        for param in self.parameters():
            parameters.append(torch.flatten(param))

        return torch.cat(parameters, dim=0)


class SurrogateNetwork_2(nn.Module):
    def __init__(self, input_dim):
        super(SurrogateNetwork_2, self).__init__()
        self.input = nn.Linear(input_dim, 64)
        self.hidden_1 = nn.Linear(64, 32)
        self.hidden_2 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)

    def forward(self, x):
        fc1 = F.relu(self.input(x))
        fc2 = F.relu(self.hidden_1(fc1))
        fc3 = F.relu(self.hidden_2(fc2))
        y_hat = F.softplus(self.output(fc3)) + 1

        return y_hat

    def parameters_to_vector(self):
        parameters = []
        for param in self.parameters():
            parameters.append(torch.flatten(param))

        return torch.cat(parameters, dim=0)


class SurrogateNetwork_3(nn.Module):
    def __init__(self, input_dim):
        super(SurrogateNetwork_3, self).__init__()
        self.input = nn.Linear(input_dim, 300)
        self.hidden_1 = nn.Linear(300, 200)
        self.hidden_2 = nn.Linear(200, 100)
        self.output = nn.Linear(100, 1)

    def forward(self, x):
        fc1 = F.relu(self.input(x))
        fc2 = F.relu(self.hidden_1(fc1))
        fc3 = F.relu(self.hidden_2(fc2))
        y_hat = F.softplus(self.output(fc3)) + 1

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
        x = F.relu(self.layers[0](x))
        for i in range(1, len(self.layers) - 1):
            x = F.relu(self.layers[i](x))

        y_hat = F.softplus(self.layers[-1](x)) + 1

        return y_hat

    def parameters_to_vector(self):
        parameters = []
        for param in self.parameters():
            parameters.append(torch.flatten(param))

        return torch.cat(parameters, dim=0)


def train_surrogate_model(params, APLs, epsilon, learning_rate=1e-2, retrain=False, current_optimizer=None,
                          current_surrogate_model=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if retrain:
        model = SurrogateNetwork_2(params.size()[1])
    else:
        if current_surrogate_model is not None:
            model = current_surrogate_model
        else:
            model = SurrogateNetwork_2(params.size()[1])

    X_train, X_val, y_train, y_val = train_test_split(params.cpu().detach().numpy(), APLs.numpy(), test_size=0.01,
                                                        random_state=42)

    model.to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if current_optimizer:
        optimizer.load_state_dict(current_optimizer)

    num_epochs = 50
    batch_size = 64

    X_train = torch.tensor(X_train, dtype=torch.float)
    y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float)
    X_val = torch.tensor(X_val, dtype=torch.float)
    y_val = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float)

    data_train = TensorDataset(X_train, y_train)
    data_train_loader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
    data_val = TensorDataset(X_val, y_val)
    data_val_loader = DataLoader(dataset=data_val, batch_size=batch_size)

    training_loss = []
    validation_loss = []

    for epoch in range(num_epochs):
        batch_loss = []

        model.train()
        for i, batch in enumerate(data_train_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)

            # if epoch == (num_epochs // 2):
            #     if np.abs(training_loss[0] - training_loss[-1]) < 1e-2:
            #         optimizer.param_groups[0]['lr'] = 2e-2

            y_hat = model(x_batch)
            loss = criterion(input=y_hat, target=y_batch) + epsilon * torch.norm(model.parameters_to_vector(), 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss.append(loss.item() / (np.var(y_train.detach().cpu().numpy()) + 0.01))

        training_loss.append(np.array(batch_loss).mean())

        model.eval()
        with torch.no_grad():
            # Test with validation data
            batch_loss = []
            for i, batch in enumerate(data_val_loader):
                x, y = batch[0].to(device), batch[1].to(device)

                y_hat = model(x)
                loss = criterion(input=y_hat, target=y)
                batch_loss.append(loss.item())

            validation_loss.append(np.array(batch_loss).mean())

        print(f'Surrogate Model: Epoch [{epoch + 1}/{num_epochs}, Loss: {np.array(batch_loss).mean():.4f}]')

    return model, optimizer.state_dict(), training_loss, validation_loss
