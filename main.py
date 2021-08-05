import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.tensorboard import SummaryWriter
from datasets import sample_2D_data, parabola, polynom_6
from sklearn.metrics import plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader, TensorDataset
import networks
from utils import save_data, get_data_loader, colormap, build_decision_tree, augment_data_with_gaussian, \
    augment_data_with_dirichlet, pred_contours
from sklearn.metrics import accuracy_score
import argparse
from PIL import Image as ImagePIL

np.random.seed(5555)
torch.random.manual_seed(5255)


def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--label',
                        required=False,
                        type=str,
                        default='1',
                        help='Label as postfix to the directory where all plots and tensorboard logs will be saved')

    parser.add_argument('--lambda_init',
                        required=False,
                        type=float,
                        default=1e-2,
                        help='Initial lambda value as regularisation term')

    parser.add_argument('--lambda_target',
                        required=False,
                        type=float,
                        default=1,
                        help='Target lambda value as regularisation term')

    parser.add_argument('--ep',
                        required=False,
                        default=300,
                        type=int,
                        help='Number of epochs, default 300 (150 warm up + 150 regularisation)')

    parser.add_argument('--batch',
                        default=32,
                        required=False,
                        help='Batch size, default 32')

    parser.add_argument('--sample',
                        type=bool,
                        required=False,
                        default=False,
                        help='Sample new data, default False')

    return parser


def snap_shot_train(data_train_loader, device, criterion, lambda_, model, epoch, path):
    y_train_predicted = []
    X_train_temp = []
    y_train_temp = []

    with torch.no_grad():
        # Test with training data
        for i, batch in enumerate(data_train_loader):
            x, y = batch[0].to(device), batch[1].to(device)
            X_train_temp.append(x)
            y_train_temp.append(y)

            y_hat = model(x)
            y_train_predicted.append(y_hat)
            loss = criterion(input=y_hat, target=y)

        X_train_temp = torch.cat(X_train_temp).cpu().numpy()
        y_train_temp = torch.cat(y_train_temp).cpu().numpy()
        y_train_predicted = torch.cat(y_train_predicted)
        y_train_predicted = torch.where(y_train_predicted > 0.5, 1, 0).detach().cpu().numpy()

    _, _, _, _ = build_decision_tree(X_train, y_train, X_train_temp, y_train_predicted, X_test,
                                     space, f"{path}/decision_tree-snapshot-epoch-{epoch}", ccp_alpha)

    xx, yy = np.linspace(space[0][0], space[0][1], 100), np.linspace(space[1][0], space[1][1], 100)
    xx, yy = np.meshgrid(xx, yy)
    Z = pred_contours(xx, yy, model)
    Z = Z.reshape(xx.shape)

    fig = plt.figure()
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
    CS = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
    # plt.colorbar()
    # plt.contour(xx, yy, Z, CS.levels, colors='k', linewidths=1.5)
    plt.scatter(*X_train_temp.T, c=colormap(y_train_predicted), edgecolors='k')
    #plt.scatter(*X_train.T, c=colormap(y_train), edgecolors='k')
    plt.xlim([space[0][0], space[0][1]])
    plt.ylim([space[1][0], space[1][1]])
    plt.title(f'Network Contourplot with Training data, $\lambda$: {lambda_}')
    # plt.plot(x_decision_fun, y_decision_fun, 'k-')
    fig.tight_layout()
    plt.savefig(f'{path}/fig_train_prediction-snapshot-epoch-{epoch}.png')
    plt.close(fig)


def train_surrogate_model(X, y, criterion, optimizer, model):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    X = torch.vstack(X)
    y = torch.tensor([y], dtype=torch.float).T

    X_train = X.cpu().detach().numpy()
    y_train = y.numpy()

    model.surrogate_network.to(device)

    num_epochs = 10
    batch_size = 32

    X_train = torch.tensor(X_train, dtype=torch.float)
    y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float)

    data_train = TensorDataset(X_train, y_train)
    data_train_loader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)

    training_loss = []

    model.surrogate_network.train()

    for epoch in range(num_epochs):
        batch_loss = []

        for i, batch in enumerate(data_train_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)

            y_hat = model.surrogate_network(x_batch)
            loss = criterion(input=y_hat, target=y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss.append(float(loss) / (np.var(y_train.cpu().detach().numpy()) + 0.01))
            # batch_loss.append(loss.item())

            del x_batch
            del y_batch

        training_loss.append(np.array(batch_loss).mean())

        print(f'Surrogate Model: Epoch [{epoch + 1}/{num_epochs}, Loss: {np.array(batch_loss).mean():.4f}]')

    del X
    del y

    return training_loss


def train(data_train_loader, writer, ccp_alpha, regulariser, strength, dim, path, args):
    device = "cuda:0" if torch.cuda.is_available() else 'cpu'

    model = networks.TreeNet(input_dim=dim)
    model.to(device)

    # Designated partial training data for APL computation
    X_apl = data_train_loader.dataset[:200][0].to(device)

    # Hypterparameters
    num_random_restarts = 25
    total_num_epochs = args.ep
    epochs_warm_up = 150
    epochs_reg = total_num_epochs - epochs_warm_up
    lambda_init = strength
    lambda_target = args.lambda_target
    lambda_ = strength

    alpha = (lambda_target / lambda_init) ** (1 / epochs_reg)

    # Objectives and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.feed_forward.parameters(), lr=1e-4)
    criterion_sr = nn.MSELoss()
    optimizer_sr = Adam(model.surrogate_network.parameters(), lr=1e-3)

    input_surrogate = []
    APLs_surrogate = []

    APLs_truth = []
    APL_predictions = []

    training_loss = []
    training_loss_without_reg = []
    surrogate_training_loss = []

    lambdas = [lambda_]

    surrogate_model_trained = False

    for i in range(num_random_restarts):
        model.reset_outer_weights()
        input_surrogate.append(model.parameters_to_vector())
        APL = model.compute_APL(X_apl, ccp_alpha)
        APLs_surrogate.append(APL)
        print(f'Random restart [{i + 1}/{num_random_restarts}]')

    for epoch in range(total_num_epochs):
        model.train()
        batch_loss = []
        batch_loss_without_reg = []

        if epoch > (epochs_warm_up - 1):

            if surrogate_model_trained:
                lambda_ = lambda_init * (alpha ** (epoch - epochs_warm_up))
                # lambda_ = lambda_target + (lambda_init - lambda_target) * ((epochs_reg - (epoch - epochs_warm_up)) / epochs_reg)
                lambdas.append(lambda_)

            input_surrogate_augmented, APLs_surrogate_augmented = augment_data_with_dirichlet(data_train_loader.dataset[:][0].to(device), input_surrogate, model, device, 200, ccp_alpha)
            model.freeze_model()
            model.surrogate_network.unfreeze_model()

            input_surrogate_augmented = input_surrogate + input_surrogate_augmented
            APLs_surrogate_augmented = APLs_surrogate + APLs_surrogate_augmented
            sr_train_loss = train_surrogate_model(input_surrogate_augmented, APLs_surrogate_augmented, criterion_sr,
                                                  optimizer_sr, model)
            surrogate_training_loss.append(sr_train_loss)

            print('Lambda: ', lambda_)

            surrogate_model_trained = True

            model.surrogate_network.freeze_model()
            model.unfreeze_model()
            model.surrogate_network.eval()

            del input_surrogate_augmented
            del APLs_surrogate_augmented
            del sr_train_loss

        if epoch % 10 == 0:  # snapshots of the resulting tree
            model.eval()
            model.freeze_model()
            snap_shot_train(data_train_loader, device, criterion, lambda_, model, epoch, path)
            model.unfreeze_model()
            model.train()

        for i, batch in enumerate(data_train_loader):

            x_batch, y_batch = batch[0].to(device), batch[1].to(device)
            y_hat = model(x_batch)

            if surrogate_model_trained:
                omega = model.compute_APL_prediction()
                loss = criterion(input=y_hat, target=y_batch) + torch.tensor(lambda_, dtype=torch.float,
                                                                             requires_grad=True) * omega
            else:
                loss = criterion(input=y_hat, target=y_batch)

            loss_without_reg = criterion(input=y_hat, target=y_batch)  # Only for plotting purpose

            batch_loss_without_reg.append(float(loss_without_reg))
            del loss_without_reg
            APL_predictions.append(model.compute_APL_prediction())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss.append(float(loss))

            # Collect weights and APLs for surrogate training
            input_surrogate.append(model.parameters_to_vector())
            APL = model.compute_APL(data_train_loader.dataset[:][0].to(device), ccp_alpha)
            APLs_surrogate.append(APL)
            APLs_truth.append(APL)

            del x_batch
            del y_batch

        print(f'Epoch: [{epoch + 1}/{total_num_epochs}, Loss: {np.array(batch_loss).mean():.4f}]')
        training_loss.append(np.array(batch_loss).mean())
        training_loss_without_reg.append(np.array(batch_loss_without_reg).mean())

    for i, _ in enumerate(surrogate_training_loss):
        for j, value in enumerate(surrogate_training_loss[i]):
            writer.add_scalar(f'Surrogate Training/Loss of surrogate training after epoch {i}', value, j)

    surrogate_training_loss = torch.tensor(surrogate_training_loss).flatten()

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].plot(range(0, len(training_loss)), training_loss)
    axs[0, 0].set_xlabel('epochs')
    axs[0, 0].set_ylabel('loss')
    axs[0, 0].grid()
    axs[0, 0].set_title(f'Training loss, $\lambda$: {lambda_}, {regulariser}')

    axs[0, 1].plot(range(0, len(training_loss_without_reg)), training_loss_without_reg)
    axs[0, 1].set_xlabel('epochs')
    axs[0, 1].set_ylabel('loss')
    axs[0, 1].grid()
    axs[0, 1].set_title('Training loss without reg')

    axs[1, 0].plot(range(0, len(surrogate_training_loss)), surrogate_training_loss)
    axs[1, 0].set_xlabel('training iterations')
    axs[1, 0].set_ylabel('loss')
    axs[1, 0].grid()
    axs[1, 0].set_title(f'Surrogate Training Loss, $\lambda$: {lambda_}, {regulariser}')

    axs[1, 1].plot(range(0, len(APLs_truth)), APLs_truth, color='y', label='true APL')
    axs[1, 1].plot(range(0, len(APL_predictions)), APL_predictions, color='g', label='predicted APL')
    axs[1, 1].set_xlabel('iterations')
    axs[1, 1].set_ylabel('node count')
    axs[1, 1].legend()
    axs[1, 1].grid()
    axs[1, 1].set_title(f'Path length estimates, $\lambda$: {lambda_}, {regulariser}')

    fig.tight_layout()
    fig.savefig(f'{path}/loss.png')
    plt.close(fig)

    fig = plt.figure()
    plt.plot(range(0, len(APLs_truth)), APLs_truth, color='y', label='true APL')
    plt.plot(range(0, len(APL_predictions)), APL_predictions, color='g', label='predicted APL')
    plt.xlabel('iterations')
    plt.ylabel('node count')
    plt.legend()
    plt.grid()
    plt.title(f'Path length estimates, $\lambda$: {lambda_}, {regulariser}')
    plt.savefig(f'{path}/APL_prediction.png')
    plt.close(fig)

    fig = plt.figure()
    plt.plot(range(0, len(lambdas)), lambdas)
    plt.xlabel('epochs')
    plt.ylabel('lambda')
    plt.grid()
    plt.title(f'Lambdas, {regulariser}')
    plt.savefig(f'{path}/lambdas.png')
    plt.close(fig)

    for i, value in enumerate(training_loss):
        writer.add_scalar('Training Loss', value, i)

    for i, value in enumerate(training_loss_without_reg):
        writer.add_scalar(f'Training Loss without Regularisation', value, i)

    for i, value in enumerate(APL_predictions):
        writer.add_scalar(f'APL Predictions: {regulariser}', value, i)

    for i, value in enumerate(surrogate_training_loss):
        writer.add_scalar(f'Surrogate Training Loss: {regulariser}', value, i)

    del input_surrogate
    del APLs_surrogate
    del criterion_sr
    del optimizer_sr

    return model, criterion, device


def init(path, tb_logs_path, strength, regulariser):
    global X_train
    global y_train
    global X_test
    global y_test
    global ccp_alpha
    global space

    num_samples, dim, space = 2000, 2, [[0, 1.5], [0, 1.5]]
    writer = SummaryWriter(log_dir=tb_logs_path)

    fun = parabola # either use paraobla, polynom_6, or create a new one
    fun_name = 'parabola'
    if args.sample:
        X, Y = sample_2D_data(num_samples, fun, space)
        save_data(X, Y, f'feed_forward_network/dataset/{fun_name}/data_{fun_name}')

    train_data_from_txt = np.loadtxt(f'dataset/{fun_name}/data_{fun_name}_train.txt')
    test_data_from_txt = np.loadtxt(f'dataset/{fun_name}/data_{fun_name}_test.txt')

    X_train, y_train = train_data_from_txt[:, :2], train_data_from_txt[:, 2]
    X_test, y_test = test_data_from_txt[:, :2], test_data_from_txt[:, 2]

    # Decision tree directly on input space
    fig_DT, fig_contour, y_hat_tree, ccp_alpha = build_decision_tree(X_train, y_train, X_train, y_train, X_test, space,
                                                                     f"{path}/decision_tree")
    acc_DT = accuracy_score(y_test, y_hat_tree)
    writer.add_text('Accuracy/Accuracy of DT', f'Accuracy with DT before reg: {acc_DT:.4f}')
    writer.add_figure(f'Decision Trees/DT before regularisation, Accuracy: {acc_DT:.4f}', fig_DT)
    writer.add_figure(f'Decision Trees/DT Contourplot before regularisation, Accuracy: {acc_DT:.4f}', fig_contour)
    plt.close(fig_DT)

    dt = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
    dt.fit(X_train, y_train)
    plot_confusion_matrix(dt, X_test, y_test)
    plt.title("Confusion Matrix Tree")
    plt.savefig(f'{path}/confusion_matrix_tree.png')
    img = ImagePIL.open(f'{path}/confusion_matrix_tree.png')
    fig = plt.figure()
    plt.imshow(img)

    writer.add_figure('Decision Trees/Confusion Matrix Before Regularisation', fig)
    plt.close(fig)

    fig = plt.figure()
    plt.scatter(*X_train.T, c=colormap(y_train), edgecolors='k')
    plt.xlim([space[0][0], space[0][1]])
    plt.ylim([space[1][0], space[1][1]])
    plt.title('Training data')

    x_decision_fun = np.linspace(space[0][0], space[0][1], 100)
    y_decision_fun = fun(x_decision_fun)

    plt.plot(x_decision_fun, y_decision_fun, 'k-')
    plt.savefig(f'{path}/samples_training_plot.png')

    # plt.show()
    writer.add_figure('Training samples', figure=fig)
    plt.close(fig)
    data_summary = f'Samples: {num_samples}  \nTraining data shape: {X_train.shape}  \nTest data shape: {X_test.shape}'
    writer.add_text('Training Data Summary', data_summary)

    # Data preparation (to Tensor then create DataLoader for batch training)
    data_train_loader, data_test_loader = get_data_loader(X_train, y_train, X_test, y_test, args.batch)

    ############# Training ######################
    print('================Training===================')
    model, criterion, device = train(data_train_loader, writer, ccp_alpha, regulariser, strength, dim, path, args)

    ############# Evaluation ######################
    print('================Test=======================')
    model.eval()
    X_train_temp = []  # Because training data are shuffled, collect them for plotting afterwards
    y_train_temp = []

    y_train_predicted = []
    y_test_predicted = []
    loss_with_train_data = []
    loss_with_test_data = []
    with torch.no_grad():
        # Test with training data
        for i, batch in enumerate(data_train_loader):
            x, y = batch[0].to(device), batch[1].to(device)
            X_train_temp.append(x)
            y_train_temp.append(y)

            y_hat = model(x)
            y_train_predicted.append(y_hat)
            loss = criterion(input=y_hat, target=y)
            loss_with_train_data.append(loss.item())

        X_train_temp = torch.cat(X_train_temp).cpu().numpy()
        y_train_temp = torch.cat(y_train_temp).cpu().numpy()
        y_train_predicted = torch.cat(y_train_predicted)
        y_train_predicted = torch.where(y_train_predicted > 0.5, 1, 0)

        # Test with test data
        for i, batch in enumerate(data_test_loader):
            x, y = batch[0].to(device), batch[1].to(device)
            y_hat = model(x)
            y_test_predicted.append(y_hat)
            loss = criterion(input=y_hat, target=y)
            loss_with_test_data.append(loss.item())

        y_test_predicted = torch.cat(y_test_predicted)
        y_test_predicted = torch.where(y_test_predicted > 0.5, 1, 0)

        ## PLOTS ##

        xx, yy = np.linspace(space[0][0], space[0][1], 100), np.linspace(space[1][0], space[1][1], 100)
        xx, yy = np.meshgrid(xx, yy)
        Z = pred_contours(xx, yy, model)
        Z = Z.reshape(xx.shape)

        # Contourplot with predicted training data
        fig = plt.figure()
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
        CS = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
        # plt.colorbar()
        # plt.contour(xx, yy, Z, CS.levels, colors='k', linewidths=1.5)
        plt.scatter(*X_train_temp.T, c=colormap(y_train_predicted), edgecolors='k')
        #plt.scatter(*X_train.T, c=colormap(y_train), edgecolors='k')
        plt.xlim([space[0][0], space[0][1]])
        plt.ylim([space[1][0], space[1][1]])
        plt.title('Network Contourplot with Training data')
        # plt.plot(x_decision_fun, y_decision_fun, 'k-')
        fig.tight_layout()
        plt.savefig(f'{path}/fig_train_prediction.png')
        writer.add_figure(f'Inference/Inference with training data, loss: {np.array(loss_with_train_data).mean()}',
                          figure=fig)
        plt.close(fig)

        # Scatterplot with test data
        fig = plt.figure()
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
        CS = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
        plt.scatter(*data_test_loader.dataset[:][0].T, c=colormap(y_test_predicted), edgecolors='k')
        #plt.scatter(*data_test_loader.dataset[:][0].T, c=colormap(y_test), edgecolors='k')
        plt.title('Network Contourplot with Test data')
        plt.xlim([space[0][0], space[0][1]])
        plt.ylim([space[1][0], space[1][1]])
        plt.savefig(f'{path}/fig_test_prediction.png')
        writer.add_figure(f'Inference/Inference with test data, loss {np.array(loss_with_test_data).mean()}',
                          figure=fig)
        plt.close(fig)

        y_train_predicted = [1 if y > 0.5 else 0 for y in y_train_predicted]
        acc_NN_train = accuracy_score(y_train_temp, y_train_predicted)
        writer.add_text('Confusion Matrices/NN with Train data', data_summary)
        writer.add_text('Accuracy/Accuracy of NN with Train data',
                        f'Accuracy of NN with train data: {acc_NN_train:.4f}')

        y_test_predicted = [1 if y > 0.5 else 0 for y in y_test_predicted]
        acc_NN_test = accuracy_score(y_test, y_test_predicted)
        writer.add_text('Confusion Matrices/NN with Test data', data_summary)
        writer.add_text('Accuracy/Accuracy of NN with Test data', f'Accuracy of NN with test data: {acc_NN_test:.4f}')

    # Decision tree after regularization

    fig_DT_reg, fig_contour, y_hat_tree, ccp_alpha = build_decision_tree(X_train, y_train, X_train_temp,
                                                                         y_train_predicted, X_test, space,
                                                                         f"{path}/decision_tree_reg", ccp_alpha)
    acc_DT_reg = accuracy_score(y_test, y_hat_tree)
    writer.add_text('Accuracy/Accuracy of DT', f'Accuracy with DT after reg: {acc_DT_reg:.4f}')
    writer.add_figure(f'Decision Trees/DT after regularisation, Accuracy: {acc_DT_reg:.4f}', fig_DT_reg)
    writer.add_figure(f'Decision Trees/DT Contourplot after regularisation, Accuracy: {acc_DT_reg:.4f}', fig_contour)
    plt.close(fig_DT_reg)

    dt = DecisionTreeClassifier(min_samples_leaf=5, ccp_alpha=ccp_alpha)
    dt.fit(X_train_temp, y_train_predicted)
    plot_confusion_matrix(dt, X_test, y_test)
    plt.title("Confusion Matrix Tree regularized NN")
    plt.savefig(f'{path}/confusion_matrix_regularized_tree.png')
    img = ImagePIL.open(f'{path}/confusion_matrix_regularized_tree.png')
    fig = plt.figure()
    plt.imshow(img)
    plt.close(fig)

    writer.add_figure('Decision Trees/Confusion Matrix After Regularisation', fig)

    plt.close(fig)

    # Final outputs

    print(f'Accuracy of NN with training data: {acc_NN_train:.4f}')
    print(f'Accuracy of NN with test data: {acc_NN_test:.4f}')
    print(f'Accuracy of NN DT before regularisation with test data: {acc_DT:.4f}')
    print(f'Accuracy of NN DT after regularisation with test data: {acc_DT_reg:.4f}')

    writer.close()
    del model


if __name__ == '__main__':

    args = parser().parse_args()
    regulariser = 'tree_reg_train'
    strength = args.lambda_init
    dir_name = f'{regulariser}_{strength}_{args.label}'

    fig_path = f'figures/{dir_name}'
    tb_logs_path = f'runs/{dir_name}'

    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    if not os.path.exists(tb_logs_path):
        os.makedirs(tb_logs_path)

    init(fig_path, tb_logs_path, strength, regulariser)
