import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from datasets import sample_2D_data, polynom_3
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from models import MLP
import tree_regularizer as tr
import decision_tree_utils as dtu
from torch.utils.data import DataLoader, TensorDataset
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score
from six import StringIO
from IPython.display import Image
import pydotplus
import argparse


# todo
def scatter_plot(plot_shape, x, y, colors, title):
    pass


# todo
def train():
    pass


# todo
def test():
    pass


def weights_to_1D_array(model_parameters):
    parameters = []
    for param in model_parameters:
        parameters.append(torch.flatten(param.data))
    weights = torch.cat(parameters)

    return weights


def main():

    num_samples = 500

    samples, labels, label_colors = sample_2D_data(num_samples)
    data_train, data_test, labels_train, labels_test = train_test_split(samples, labels, test_size=0.33,
                                                                        random_state=42)

    fig, ax = plt.subplots()
    plt.xlim([0, 1.5])
    plt.ylim([0, 1.5])
    ax.scatter(data_train[:, 0], data_train[:, 1], c=label_colors)
    ax.set_title('Training data')

    xx = np.linspace(0, 1.5, 50)
    yy = polynom_3(xx)

    ax.plot(xx, yy, 'k-')
    plt.savefig('figures/samples_training_plot.png')

    # plt.show()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = MLP(data_train.shape[1], 500)
    model.to(device)

    # Hypterparameters
    batch_size = args.batch
    input_batch_size_surrogate_training = args.sb
    regularization_strength = args.rs
    learning_rate = args.lr

    # Objectives and Optimizer
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    surrogate_model = None
    surrogate_model_trained = False
    criterion_tr = tr.Tree_Regularizer(regularization_strength)

    # Data perparation
    data_train = torch.tensor(data_train, dtype=torch.float32)
    labels_train = torch.tensor(labels_train.reshape(-1, 1), dtype=torch.float32)
    data_test = torch.tensor(data_test, dtype=torch.float32)
    labels_test = torch.tensor(labels_test.reshape(-1, 1), dtype=torch.float32)

    train_data = TensorDataset(data_train, labels_train)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size)

    test_data = TensorDataset(data_test, labels_test)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size)

    print('================Training===================')

    num_epochs = 250
    input_data_surrogate_training = []
    APLs = []
    training_loss = []
    loss_surrogate_training = []
    # todo --> try to use PyTorch-Lightning ...
    for epoch in range(num_epochs):
        model.train()
        running_loss = []

        if input_batch_size_surrogate_training > 0:
            for i, batch in enumerate(train_loader):
                x_batch, y_batch = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()
                y_hat = model(x_batch)

                parameters = []
                for param in model.parameters():
                    parameters.append(torch.flatten(param.data))
                weights = torch.cat(parameters)

                if surrogate_model_trained:
                    loss = criterion_tr(input=y_batch, target=y_hat, weights=weights, model=surrogate_model)
                else:
                    loss = criterion(input=y_batch, target=y_hat)

                loss.backward()
                optimizer.step()

                running_loss.append(loss.item())

            # todo stack model paramaters
            parameters = []
            for param in model.parameters():
                parameters.append(torch.flatten(param.data))

            input_data_surrogate_training.append(torch.cat(parameters))

            average_path_length = dtu.average_path_length(X_train=train_loader.dataset[:][0].to(device),
                                                          X_test=test_loader.dataset[:][0],
                                                          y_test=test_loader.dataset[:][1], model=model)
            APLs.append(average_path_length)

            print(f'Epoch: {epoch + 1}/{num_epochs}, loss: {np.array(running_loss).mean():.4f}')
            training_loss.append(np.array(running_loss).mean())

            input_batch_size_surrogate_training -= 1
        else:
            print('================Training surrogate model===================')
            input_data_surrogate_training = torch.vstack(input_data_surrogate_training)
            APLs = torch.tensor([APLs], dtype=torch.float32).T

            if args.sw:
                if surrogate_model_trained:
                    surrogate_model, sr_loss = tr.train_surrogate_model(input_data_surrogate_training, APLs,
                                                                        args.epsilon, learning_rate=args.lr_sr,
                                                                        current_surrogate_model=surrogate_model)
                else:
                    surrogate_model, sr_loss = tr.train_surrogate_model(input_data_surrogate_training, APLs,
                                                                        args.epsilon, learning_rate=args.lr_sr)
                    surrogate_model_trained = True
            else:
                surrogate_model, sr_loss = tr.train_surrogate_model(input_data_surrogate_training, APLs,
                                                                    args.epsilon,learning_rate=args.lr_sr, retrain=True)
                surrogate_model_trained = True



            loss_surrogate_training.append(sr_loss)

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

                loss = criterion_tr(y_batch, y_hat, weights, surrogate_model)
                loss.backward()
                optimizer.step()

                running_loss.append(loss.item())

            input_data_surrogate_training = []
            APLs = []

            # print(f'Epoch: {epoch+1}/{num_epochs}, step: {i}/{batch_iteration}, loss: {loss.item():.4f}')
            print('================After Training surrogate model===================')
            print(f'Epoch: {epoch + 1}/{num_epochs}, loss: {np.array(running_loss).mean():.4f}')
            training_loss.append(np.array(running_loss).mean())

            input_batch_size_surrogate_training = args.sb

    fig, ax = plt.subplots()
    ax.plot(np.linspace(0, num_epochs, len(training_loss)), training_loss)
    ax.set_title('Training Loss')
    plt.savefig('figures/training_loss.png')

    num_plots = len(loss_surrogate_training)
    fig, ax = plt.subplots(num_plots, figsize=(8, 20))
    for i in range(len(loss_surrogate_training)):
        ax[i].plot(np.linspace(0, 250, len(loss_surrogate_training[i])), loss_surrogate_training[i])
        ax[i].set_title(f'{i} Surrogate Training Loss')
    fig.tight_layout()
    plt.savefig('figures/surrogate_training_loss.png')


    print('================Test===================')
    model.eval()
    y_predicted = []
    batch_iteration = len(train_loader.dataset) // batch_size
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            x_train, y_train = batch[0].to(device), batch[1].to(device)
            y_prediction = model(x_train)
            y_predicted.append(y_prediction)

            parameters = []
            for param in model.parameters():
                parameters.append(torch.flatten(param.data))
            weights = torch.cat(parameters)

            loss = criterion_tr(input=y_prediction, target=y_train, weights=weights, model=surrogate_model)

            print(f'Batch: {i}/{batch_iteration}, loss: {loss.item()}')

        y_predicted = torch.cat(y_predicted)

        label_colors_true = ['r' if train_loader.dataset[i][1] == 1 else 'b' for i in
                             range(len(train_loader.dataset[:][1]))]
        label_colors_predicted = ['r' if y_predicted[i] > 0.5 else 'b' for i in range(len(y_predicted))]

        fig, ax = plt.subplots(2)
        ax[0].set_xlim([0, 1.5])
        ax[0].set_ylim([0, 1.5])
        ax[1].set_xlim([0, 1.5])
        ax[1].set_ylim([0, 1.5])
        ax[0].scatter(data_train[:, 0], data_train[:, 1], c=label_colors_true)
        ax[0].set_title('Ground truth')
        ax[1].scatter(data_train[:, 0], data_train[:, 1], c=label_colors_predicted)
        ax[1].set_title('Predicted')

        xx = np.linspace(0, 1.5, 50)
        yy = polynom_3(xx)

        ax[0].plot(xx, yy, 'k-')
        ax[1].plot(xx, yy, 'k-')

        plt.savefig('figures/training_samples_prediction_plot.png')



    # Decision tree directly on input space
    final_decision_tree = DecisionTreeClassifier(min_samples_leaf=25, ccp_alpha=0.001)
    final_decision_tree.fit(data_train, labels_train)
    #final_decision_tree = dtu.post_pruning(data_train, labels_train, data_test, labels_test, final_decision_tree)

    y_predicted_tree = final_decision_tree.predict(data_test)
    print(f'Accuracy Decision Tree directly on input data: {accuracy_score(labels_test, y_predicted_tree):.4f}')

    # Export graph
    dot_data = StringIO()
    export_graphviz(final_decision_tree, out_file=dot_data, filled=True, rounded=True, special_characters=True,
                    feature_names=['positive', 'negative'],
                    class_names=['0', '1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('figures/direct_decision_tree_2.png')
    Image(graph.create_png())

    # Decision tree after regularization
    final_decision_tree = DecisionTreeClassifier(min_samples_split=25, ccp_alpha=0.001)
    y_predicted = y_predicted.to('cpu').detach().numpy()
    y_predicted = [1 if y_predicted[i] > 0.5 else 0 for i in range(len(y_predicted))]
    final_decision_tree.fit(data_train, y_predicted)
    #final_decision_tree = dtu.post_pruning(data_train, y_predicted, data_test, labels_test, final_decision_tree)

    y_predicted_tree = final_decision_tree.predict(data_test)
    print(f'Accuracy Decision Tree after regularization: {accuracy_score(labels_test, y_predicted_tree):.4f}')

    # Export graph
    dot_data = StringIO()
    export_graphviz(final_decision_tree, out_file=dot_data, filled=True, rounded=True, special_characters=True,
                    feature_names=['positive', 'negative'],
                    class_names=['0', '1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('figures/decision_tree_after_regularization.png')
    Image(graph.create_png())

    #breakpoint()

    print("Differences: ", np.sum(np.abs(np.array(labels_train) - np.array(y_predicted))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch',
                        type=int,
                        default=64,
                        required=False,
                        help='Batch size, default 64')

    parser.add_argument('--rs',
                        type=float,
                        required=False,
                        default=10.0,
                        help='Regularization strength for the objective, default 10')

    parser.add_argument('--epsilon',
                        type=int,
                        default=10,
                        required=False,
                        help='Regularization strength for the surrogate training, default 10')

    parser.add_argument('--lr',
                        type=float,
                        required=False,
                        default=0.001,
                        help='Learning rate, default 0.001')

    parser.add_argument('--lr_sr',
                        type=float,
                        required=False,
                        default=0.001,
                        help='Learning rate, default 0.001')

    parser.add_argument('--rt',
                        type=bool,
                        required=False,
                        default=True,
                        help='Retrain network, default True')

    parser.add_argument('--sw',
                        type=bool,
                        required=False,
                        default=True,
                        help='Surrogate training with saved weights, default True')

    parser.add_argument('--sb',
                        type=int,
                        required=False,
                        default=25,
                        help='Input size for surrogate training, default 25')

    args = parser.parse_args()

    main()
