import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from datasets import gen_synthetic_dataset
from sklearn.metrics import plot_confusion_matrix, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader, TensorDataset
import networks
from utils import get_data_loader, colormap, build_decision_tree, augment_data_with_dirichlet, pred_contours
import argparse
from PIL import Image as ImagePIL

np.random.seed(5555)
torch.random.manual_seed(5255)


def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--label',
                        required=False,
                        type=str,
                        default='train',
                        help='Label as postfix to the directory where all plots and tensorboard logs will be saved')

    parser.add_argument('--lambda_init',
                        required=False,
                        type=float,
                        default=1e-2,
                        help='Initial lambda value as regularisation term')

    parser.add_argument('--lambda_target',
                        required=False,
                        type=float,
                        default=2e-1,
                        help='Target lambda value as regularisation term')

    parser.add_argument('--ep',
                        required=False,
                        default=450,
                        type=int,
                        help='Number of epochs, default 300 (150 warm up + 150 regularisation)')

    parser.add_argument('--batch',
                        default=100,
                        required=False,
                        help='Batch size, default 32')

    return parser


def snap_shot_train(data_train_loader, criterion, lambda_, ccp_alpha, model, epoch, path):
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
            loss = criterion(input=y_hat, target=y[:, -1, :])

        X_train_temp = torch.cat(X_train_temp).cpu().numpy()
        y_train_temp = torch.cat(y_train_temp).cpu().numpy()
        y_train_predicted = torch.cat(y_train_predicted)
        y_train_predicted = torch.where(y_train_predicted > 0.5, 1, 0).detach().cpu().numpy()

    _, _, _ = build_decision_tree(X_train_temp[:, -1, :], y_train_predicted, X_test[:, -1, :], f"{path}/decision_tree-snapshot-epoch-{epoch}", ccp_alpha)



def train_surrogate_model(X, y, criterion, optimizer, model):

    X_train = torch.vstack(X).detach()
    y_train = torch.tensor([y], dtype=torch.float).T.to(device)

    model.surrogate_network.to(device)

    num_epochs = 10
    batch_size = 100

    data_train = TensorDataset(X_train, y_train)
    data_train_loader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)

    training_loss = []

    model.surrogate_network.train()

    for epoch in range(num_epochs):
        batch_loss = []

        for (x, y) in data_train_loader:
            y_hat = model.surrogate_network(x)
            loss = criterion(input=y_hat, target=y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss.append(loss.item() / (torch.var(y_train).item() + 0.01))

        training_loss.append(np.array(batch_loss).mean())

        print(f'Surrogate Model: Epoch [{epoch + 1}/{num_epochs}, Loss: {np.array(batch_loss).mean():.4f}]')

    del X
    del y

    return training_loss


def train(data_train_loader, data_val_loader, writer, ccp_alpha, path):

    model = networks.TreeGRU(input_size, 25, 1)
    model.to(device)

    model_states_dict = []

    # Hypterparameters
    num_random_restarts = 25
    total_num_epochs = args.ep
    epochs_warm_up = 150
    epochs_reg = total_num_epochs - epochs_warm_up
    lambda_init = args.lambda_init
    lambda_target = args.lambda_target
    lambda_ = lambda_init

    alpha = (lambda_target / lambda_init) ** (1 / epochs_reg)
    cooling_fun = lambda k: lambda_target + (lambda_init - lambda_target) * (1 / (1 + np.exp(((-6 * np.log((np.abs(lambda_init - lambda_target))) / epochs_reg) * (k - epochs_reg / 2)))))

    # Objectives and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.gru_rnn.parameters(), lr=2e-3)
    criterion_sr = nn.MSELoss()
    optimizer_sr = Adam(model.surrogate_network.parameters(), lr=1e-3, weight_decay=1e-5)

    input_surrogate = []
    APLs_surrogate = []

    APLs_truth = []
    APL_predictions = []

    training_loss = []
    val_loss = []
    training_auc = []
    training_loss_without_reg = []
    surrogate_training_loss = []

    lambdas = [lambda_]

    surrogate_model_trained = False

    input_data = data_train_loader.dataset[:][0]

    for i in range(num_random_restarts):
        model.reset_outer_weights()
        input_surrogate.append(model.parameters_to_vector())
        APL = model.compute_APL(input_data, ccp_alpha)
        APLs_surrogate.append(APL)
        print(f'Random restart [{i + 1}/{num_random_restarts}]')

    for epoch in range(total_num_epochs):
        model.train()
        batch_loss_train = []
        batch_loss_val = []
        batch_loss_without_reg = []
        batch_auc = []


        if epoch > (epochs_warm_up - 1):

            if surrogate_model_trained:
                #lambda_ = lambda_init * (alpha ** (epoch - epochs_warm_up))
                # lambda_ = lambda_target + (lambda_init - lambda_target) * ((epochs_reg - (epoch - epochs_warm_up)) / epochs_reg)
                lambda_ = cooling_fun(epoch - epochs_warm_up)
                lambdas.append(lambda_)

            input_surrogate_augmented, APLs_surrogate_augmented = augment_data_with_dirichlet(input_data, input_surrogate, model, device, 300, ccp_alpha)
            model.freeze_model()
            model.surrogate_network.unfreeze_model()

            input_surrogate_augmented = input_surrogate + input_surrogate_augmented
            APLs_surrogate_augmented = APLs_surrogate + APLs_surrogate_augmented
            sr_train_loss = train_surrogate_model(input_surrogate_augmented, APLs_surrogate_augmented, criterion_sr, optimizer_sr, model)
            surrogate_training_loss.append(sr_train_loss)

            print('Lambda: ', lambda_)

            surrogate_model_trained = True

            model.surrogate_network.freeze_model()
            model.unfreeze_model()
            model.surrogate_network.eval()

            del input_surrogate_augmented
            del APLs_surrogate_augmented
            del sr_train_loss

        for (x, y) in data_train_loader:

            y_hat = model(x)

            if surrogate_model_trained:
                omega = model.compute_APL_prediction()
                loss = criterion(input=y_hat, target=y[:, -1, :]) + lambda_ * omega
            else:
                loss = criterion(input=y_hat, target=y[:, -1, :])

            loss_without_reg = criterion(input=y_hat, target=y[:, -1, :])  # Only for plotting purpose

            batch_loss_without_reg.append(float(loss_without_reg))
            del loss_without_reg

            if surrogate_model_trained:
                APL_predictions.append(model.compute_APL_prediction())
            else:
                APL_predictions.append(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss_train.append(loss.item())

            # Collect weights and APLs for surrogate training
            input_surrogate.append(model.parameters_to_vector())
            APL = model.compute_APL(input_data, ccp_alpha)
            APLs_surrogate.append(APL)
            APLs_truth.append(APL)

            del x, y

        if epoch % 10 == 0:  # snapshots of the resulting tree
            torch.save(model.state_dict(), f'models/model_snapshot_{epoch}.pth')
            model.eval()
            model.freeze_model()
            snap_shot_train(data_train_loader, criterion, lambda_, ccp_alpha, model, epoch, path)
            model.unfreeze_model()
            model.train()

        # Validation
        model.eval()
        for (x, y) in data_val_loader:
            y_hat = model(x)
            loss = criterion(input=y_hat, target=y[:, -1, :])
            batch_loss_val.append(float(loss))
            batch_auc.append(roc_auc_score(y[:, -1, :].detach().cpu().numpy(), y_hat.detach().cpu().numpy()))

            del x, y

        print(f'Epoch: [{epoch + 1}/{total_num_epochs}, Loss: {np.array(batch_loss_train).mean():.4f}]')
        training_loss.append(np.array(batch_loss_train).mean())
        val_loss.append(np.array(batch_loss_val).mean())
        training_auc.append(np.array(batch_auc).mean())
        training_loss_without_reg.append(np.array(batch_loss_without_reg).mean())

        # for (epoch, lambda_, model_state) in model_states_dict:
        #     model = networks.TreeGRU(input_size, 25, 1)
        #     model.load_state_dict(model_state)
        #     model.eval()
        #     snap_shot_train(data_train_loader, criterion, lambda_, ccp_alpha, model, epoch, path)

    for i, _ in enumerate(surrogate_training_loss):
        for j, value in enumerate(surrogate_training_loss[i]):
            writer.add_scalar(f'Surrogate Training/Loss of surrogate training after epoch {i}', value, j)


    # PLOTS
    surrogate_training_loss = torch.tensor(surrogate_training_loss).flatten()

    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    axs[0, 0].plot(range(0, len(training_loss)), training_loss)
    axs[0, 0].set_xlabel('epochs')
    axs[0, 0].set_ylabel('loss')
    axs[0, 0].grid()
    axs[0, 0].set_title(f'Training loss')

    axs[0, 1].plot(range(0, len(training_loss_without_reg)), training_loss_without_reg, label='Training loss')
    axs[0, 1].plot(range(0, len(val_loss)), val_loss, label='Validation loss')
    axs[0, 1].set_xlabel('epochs')
    axs[0, 1].set_ylabel('loss')
    axs[0, 1].legend()
    axs[0, 1].grid()
    axs[0, 1].set_title('Training loss without reg')

    axs[1, 0].plot(range(0, len(surrogate_training_loss)), surrogate_training_loss)
    axs[1, 0].set_xlabel('epochs')
    axs[1, 0].set_ylabel('loss')
    axs[1, 0].grid()
    axs[1, 0].set_title(f'Surrogate Training Loss')

    axs[1, 1].plot(range(0, len(APLs_truth)), APLs_truth, color='y', label='true APL')
    axs[1, 1].plot(range(0, len(APL_predictions)), APL_predictions, color='g', label='predicted APL $\hat{\Omega}(W)$')
    axs[1, 1].set_xlabel('iterations')
    axs[1, 1].set_ylabel('node count')
    axs[1, 1].legend()
    axs[1, 1].grid()
    axs[1, 1].set_title(f'Path length estimates')

    axs[0, 2].plot(range(0, len(training_auc)), training_auc, color='r')
    axs[0, 2].set_xlabel('epochs')
    axs[0, 2].set_ylabel('AUC')
    axs[0, 2].grid()
    axs[0, 2].set_title(f'AUC')

    fig.delaxes(axs[1, 2])

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
    plt.title(f'Path length estimates')
    plt.savefig(f'{path}/APL_prediction.png')
    plt.close(fig)

    fig = plt.figure()
    plt.plot(range(0, len(lambdas)), lambdas)
    plt.xlabel('epochs')
    plt.ylabel('lambda')
    plt.grid()
    plt.title(f'Lambda curve')
    plt.savefig(f'{path}/lambdas.png')
    plt.close(fig)

    for i, value in enumerate(training_loss):
        writer.add_scalar('Training Loss', value, i)

    for i, value in enumerate(training_loss_without_reg):
        writer.add_scalar(f'Training Loss without Regularisation', value, i)

    for i, value in enumerate(APL_predictions):
        writer.add_scalar(f'APL Predictions', value, i)

    for i, value in enumerate(surrogate_training_loss):
        writer.add_scalar(f'Surrogate Training Loss', value, i)

    del input_surrogate
    del APLs_surrogate
    del criterion_sr
    del optimizer_sr

    return model, criterion


def init(path, tb_logs_path):
    global X_train
    global y_train
    global X_test
    global y_test
    global ccp_alpha

    writer = SummaryWriter(log_dir=tb_logs_path)

    # train_data_from_txt = np.loadtxt(f'dataset/{dataset}/data_{dataset}_train.txt')
    # test_data_from_txt = np.loadtxt(f'dataset/{dataset}/data_{dataset}_test.txt')
    # val_data_from_txt = np.loadtxt(f'dataset/{dataset}/data_{dataset}_val.txt')
    #
    # X_train, y_train = train_data_from_txt[:, :2], train_data_from_txt[:, 2]
    # X_test, y_test = test_data_from_txt[:, :2], test_data_from_txt[:, 2]
    # X_val, y_val = val_data_from_txt[:, :2], val_data_from_txt[:, 2]

    # data split 70/15/15 ratio
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    # Decision tree directly on input space
    fig_DT, y_hat_tree, ccp_alpha = build_decision_tree(X_train[:, -1, :], y_train[:, -1, :], X_test[:, -1, :], f"{path}/decision_tree")

    auc_DT = roc_auc_score(y_test[:, -1, :], y_hat_tree)
    writer.add_text('AUC/AUC of DT', f'AUC of DT before reg: {auc_DT:.4f}')
    writer.add_figure(f'Decision Trees/DT before regularisation, AUC: {auc_DT:.4f}', fig_DT)
    plt.close(fig_DT)

    dt = DecisionTreeClassifier(min_samples_leaf=5, random_state=42)
    dt.fit(X_train[:, -1, :], y_train[:, -1, :])
    plot_confusion_matrix(dt, X_test[:, -1, :], y_test[:, -1, :])
    plt.title("Confusion Matrix Tree")
    plt.savefig(f'{path}/confusion_matrix_tree.png')
    img = ImagePIL.open(f'{path}/confusion_matrix_tree.png')
    fig = plt.figure()
    plt.imshow(img)

    writer.add_figure('Decision Trees/Confusion Matrix Before Regularisation', fig)
    plt.close(fig)

    data_summary = f'Samples: {num_samples}  \nTraining data shape: {X_train.shape}  \nTest data shape: {X_test.shape}'
    writer.add_text('Training Data Summary', data_summary)

    # Data preparation (to Tensor then create DataLoader for batch training)
    data_train_loader, data_test_loader, data_val_loader = get_data_loader(X_train, y_train, X_test, y_test, X_val, y_val, torch.float, torch.float, args.batch)

    ############# Training ######################
    print('Training'.center(len('Training') + 2).center(30, '='))
    model, criterion = train(data_train_loader, data_val_loader, writer, ccp_alpha, path)

    ############# Evaluation #####################
    print('Test'.center(len('Test') + 2).center(30, '='))
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
            loss = criterion(input=y_hat, target=y[:, -1, :])
            loss_with_train_data.append(loss.item())

        X_train_temp = torch.cat(X_train_temp).cpu().numpy()
        y_train_temp = torch.cat(y_train_temp).cpu().numpy()
        y_train_predicted = torch.cat(y_train_predicted)
        y_train_predicted = torch.where(y_train_predicted > 0.5, 1, 0).detach().cpu().numpy()

        # Test with test data
        for i, batch in enumerate(data_test_loader):
            x, y = batch[0].to(device), batch[1].to(device)
            y_hat = model(x)
            y_test_predicted.append(y_hat)
            loss = criterion(input=y_hat, target=y[:, -1, :])
            loss_with_test_data.append(loss.item())

        y_test_predicted = torch.cat(y_test_predicted)
        y_test_predicted = torch.where(y_test_predicted > 0.5, 1, 0).detach().cpu().numpy()

        auc_NN_train = roc_auc_score(y_train_temp[:, -1, :], y_train_predicted)
        writer.add_text('Confusion Matrices/NN with Train data', data_summary)
        writer.add_text('AUC/AUC of NN with Train data', f'AUC of NN with train data: {auc_NN_train:.4f}')

        auc_NN_test = roc_auc_score(y_test[:, -1, :], y_test_predicted)
        writer.add_text('Confusion Matrices/NN with Test data', data_summary)
        writer.add_text('AUC/AUC of NN with Test data', f'AUC of NN with test data: {auc_NN_test:.4f}')

    # Decision tree after regularization

    fig_DT_reg, y_hat_tree, ccp_alpha = build_decision_tree(X_train_temp[:, -1, :], y_train_predicted, X_test[:, -1, :], f"{path}/decision_tree_reg", ccp_alpha)
    auc_DT_reg = roc_auc_score(y_test[:, -1, :], y_hat_tree)
    writer.add_text('AUC/AUC of DT', f'AUC with DT after reg: {auc_DT_reg:.4f}')
    writer.add_figure(f'Decision Trees/DT after regularisation, AUC: {auc_DT_reg:.4f}', fig_DT_reg)
    plt.close(fig_DT_reg)

    dt = DecisionTreeClassifier(min_samples_leaf=5, random_state=42)
    dt.fit(X_train_temp[:, -1, :], y_train_predicted)
    plot_confusion_matrix(dt, X_test[:, -1, :], y_test[:, -1, :])
    plt.title("Confusion Matrix Tree regularized NN")
    plt.savefig(f'{path}/confusion_matrix_regularized_tree.png')
    img = ImagePIL.open(f'{path}/confusion_matrix_regularized_tree.png')
    fig = plt.figure()
    plt.imshow(img)
    plt.close(fig)

    writer.add_figure('Decision Trees/Confusion Matrix After Regularisation', fig)

    plt.close(fig)

    # Final outputs

    print(f'AUC of NN with training data: {auc_NN_train:.4f}')
    print(f'AUC of NN with test data: {auc_NN_test:.4f}')
    print(f'AUC of NN DT before regularisation with test data: {auc_DT:.4f}')
    print(f'AUC of NN DT after regularisation with test data: {auc_DT_reg:.4f}')

    writer.close()
    del model


if __name__ == '__main__':

    device = "cuda:0" if torch.cuda.is_available() else 'cpu'

    args = parser().parse_args()
    input_size = 14
    num_samples = 2000
    dataset = 'signal_noise_hmm'

    X, Y = gen_synthetic_dataset(num_samples, 50)

    dir_name = f'tree_reg_train_{args.lambda_init}_{args.lambda_target}_{args.label}'

    fig_path = f'figures/{dir_name}'
    tb_logs_path = f'runs/{dir_name}'

    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    if not os.path.exists(tb_logs_path):
        os.makedirs(tb_logs_path)

    init(fig_path, tb_logs_path)
