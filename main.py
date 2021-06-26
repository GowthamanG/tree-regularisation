import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from datasets import sample_2D_data, parabola, polynom_3, polynom_6
from sklearn.metrics import confusion_matrix
import networks
import tree_regularisation as tr
import decision_tree_utils as dtu
from utils import save_data, get_data_loader, colormap, build_decision_tree, augment_data, pred_contours
from sklearn.metrics import accuracy_score
import argparse


def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ep',
                        type=int,
                        required=False,
                        default=100,
                        help='Number of epochs, default 150')

    parser.add_argument('--batch',
                        type=int,
                        default=32,
                        required=False,
                        help='Batch size, default 100')

    parser.add_argument('--lr',
                        type=float,
                        required=False,
                        default=1e-2,
                        help='Learning rate, default 1e-2')

    parser.add_argument('--lr_sr',
                        type=float,
                        required=False,
                        default=1e-2,
                        help='Learning rate, default 1e-2')

    parser.add_argument('--rs',
                        type=float,
                        required=False,
                        default=1e-1,
                        help='Regularization strength for the objective, default 1e-1')

    parser.add_argument('--epsilon',
                        type=float,
                        default=0.0,
                        required=False,
                        help='Regularization strength for the surrogate training, default 1e-1')

    parser.add_argument('--rt',
                        type=bool,
                        required=False,
                        default=True,
                        help='(Retrain network, default True)')

    parser.add_argument('--sw',
                        type=bool,
                        required=False,
                        default=True,
                        help='Surrogate training with saved weights, default True')

    parser.add_argument('--agg',
                        type=bool,
                        required=False,
                        default=True,
                        help='Surrogate training with input aggregation, default True')

    parser.add_argument('--sr_batch',
                        type=int,
                        required=False,
                        default=1,
                        help='Input size for surrogate training, default 25')

    parser.add_argument('--save',
                        type=bool,
                        required=False,
                        default=False,
                        help='Sample new data, default False')

    return parser


def train_surrogate_model(W, APLs, learning_rate, epsilon, optimizer=None, model=None):
    X = torch.vstack(W)
    y = torch.tensor([APLs], dtype=torch.float).T

    surrogate_model, optimizer_state_dict, sr_loss = tr.train_surrogate_model(
                                                    params=X,
                                                    APLs=y,
                                                    epsilon=epsilon,
                                                    learning_rate=learning_rate,
                                                    current_optimizer=optimizer,
                                                    current_surrogate_model=model)

    return surrogate_model, optimizer_state_dict, sr_loss


def train_surrogate_model_with_aggregation(W, APLs, learning_rate, epsilon, optimizer=None, model=None):
    X = torch.vstack(W)
    y = torch.tensor([APLs], dtype=torch.float).T

    surrogate_model, optimizer_state_dict, sr_loss = tr.train_surrogate_model(
                                                    params=X,
                                                    APLs=y,
                                                    epsilon=epsilon,
                                                    learning_rate=learning_rate,
                                                    current_optimizer=optimizer,
                                                    current_surrogate_model=model)

    return surrogate_model, optimizer_state_dict, sr_loss


def train(data_train_loader, data_test_loader, writer, ccp_alpha, regulariser, strength, dim, path, args):

    model = networks.Net1(input_dim=dim)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    # Hypterparameters
    num_epochs = args.ep
    batch_size = args.batch
    regularization_strength = strength
    learning_rate = args.lr

    # Objectives and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    surrogate_model = None
    optimizer_surrogate_model = None
    surrogate_model_trained = False

    num_iter = args.sr_batch
    input_data_st = []
    APLs = []
    training_loss = []
    validation_loss = []
    loss_surrogate_training = []

    loss_without_reg_plot = []
    omega_plot = []

    for epoch in range(num_epochs):
        model.train()
        batch_loss = []
        batch_loss_without_reg = []

        # Train surrogate model after every 'num_iter'
        if num_iter == 0:
            fig = plt.figure()
            plt.hist(APLs)
            plt.title(f'Histogram of APLs after epoch {epoch + 1}')
            plt.xlabel('APLs')
            writer.add_figure(f'APL Histogram/APLs Histogram after epoch: {epoch + 1}', fig)
            plt.close(fig)

            ######## Surrogate Training with/without aggregation, train/retrain surrogate model##############

            input_data_st_augmented, APLs_augmented = augment_data(data_train_loader.dataset[:][0].to(device),
                                                                   data_test_loader.dataset[:][0],
                                                                   data_test_loader.dataset[:][1], model, device,
                                                                   len(APLs), ccp_alpha)

            if args.agg:
                # Train surrogate model without input (weights) aggregation
                if args.sw:
                    surrogate_model, optimizer_surrogate_model, sr_loss = train_surrogate_model_with_aggregation(
                        input_data_st + input_data_st_augmented,
                        APLs + APLs_augmented,
                        args.lr_sr,
                        args.epsilon,
                        optimizer=optimizer_surrogate_model,
                        model=surrogate_model)


                else:
                    surrogate_model, optimizer_surrogate_model, sr_loss = train_surrogate_model_with_aggregation(
                        input_data_st + input_data_st_augmented,
                        APLs + APLs_augmented,
                        args.lr_sr,
                        args.epsilon)

            else:
                # Train surrogate model without input (weights) aggregation
                if args.sw:
                    surrogate_model, optimizer_surrogate_model, sr_loss = train_surrogate_model(
                        input_data_st + input_data_st_augmented,
                        APLs + APLs_augmented,
                        args.lr_sr,
                        args.epsilon,
                        optimizer=optimizer_surrogate_model,
                        model=surrogate_model)
                else:
                    surrogate_model, optimizer_surrogate_model, sr_loss = train_surrogate_model(
                        input_data_st + input_data_st_augmented,
                        APLs + APLs_augmented,
                        args.lr_sr,
                        args.epsilon)

                input_data_st = []
                APLs = []

            num_iter = args.sr_batch
            surrogate_model_trained = True
            loss_surrogate_training.append(sr_loss)

        # Training loop of the first network
        for i, batch in enumerate(data_train_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()
            y_hat = model(x_batch)

            if surrogate_model_trained:
                surrogate_model.zero_grad()
                surrogate_model.eval()

                omega = lambda x: {
                    'l1': torch.norm(x, 1),
                    'l2': torch.norm(x, 2),
                    'tr': surrogate_model(x)
                }

                loss = criterion(input=y_hat, target=y_batch) + regularization_strength * \
                       omega(model.parameters_to_vector())[regulariser]

                loss_without_reg = criterion(input=y_hat, target=y_batch)
                batch_loss_without_reg.append(loss_without_reg.item())
                omega_plot.append(omega(model.parameters_to_vector())[regulariser])

                loss.backward()

            else:
                loss = criterion(input=y_hat, target=y_batch)
                batch_loss_without_reg.append(loss.item())
                loss.backward()

            optimizer.step()
            batch_loss.append(loss.item())

            # Stack model parameters and APLs after every epoch for surrogate training
            input_data_st.append(model.parameters_to_vector())
            average_path_length = dtu.average_path_length(X_train=data_train_loader.dataset[:][0].to(device),
                                                          X_test=data_test_loader.dataset[:][0],
                                                          y_test=data_test_loader.dataset[:][1],
                                                          model=model,
                                                          ccp_alpha=ccp_alpha)
            APLs.append(average_path_length)

        print(f'Epoch: {epoch + 1}/{num_epochs}, loss: {np.array(batch_loss).mean():.4f}')
        training_loss.append(np.array(batch_loss).mean())
        loss_without_reg_plot.append(np.array(batch_loss_without_reg).mean())

        num_iter -= 1

    for i, value in enumerate(training_loss):
        writer.add_scalar('Training Loss', value, i)

    plt.figure()
    plt.plot(range(0, len(training_loss)), training_loss)
    plt.title(f'Training loss, $\lambda: {regularization_strength}, {regulariser}')
    plt.savefig(f'{path}/loss.png')

    for i, value in enumerate(loss_without_reg_plot):
        writer.add_scalar(f'Loss without regularisation', value, i)

    for i, value in enumerate(omega_plot):
        writer.add_scalar(f'Omega Values: {regulariser}', value, i)

    for i in range(len(loss_surrogate_training)):
        for j in range(len(loss_surrogate_training[i])):
            writer.add_scalar(f'Surrogate Training/Loss of surrogate training after epoch {i}',
                              loss_surrogate_training[i][j], j)

    del input_data_st
    del APLs
    del surrogate_model

    return model, criterion, device



def init(path, strength, regulariser):

    num_samples, dim, space = 2000, 2, [[0, 1.5], [0, 1.5]]
    writer = SummaryWriter()

    fun = parabola # either use paraobla, polynom_3, polynom_3 or create a new one
    if args.save:
        X, Y = sample_2D_data(num_samples, fun, space)
        save_data(X, Y, 'dataset/data_parabola')

    train_data_from_txt = np.loadtxt('dataset/data_parabola_train.txt')
    test_data_from_txt = np.loadtxt('dataset/data_parabola_test.txt')
    X_train = train_data_from_txt[:, :2]
    y_train = train_data_from_txt[:, 2]
    X_test = test_data_from_txt[:, :2]
    y_test = test_data_from_txt[:, 2]

    # Decision tree directly on input space

    fig_DT, fig_contour, y_hat_tree, ccp_alpha = build_decision_tree(X_train, y_train, X_test, y_test, space, f"{path}/decision_tree")
    acc_DT = accuracy_score(y_test, y_hat_tree)
    tn, fp, fn, tp = confusion_matrix(y_test, y_hat_tree).ravel()
    data_summary = f'DT before reg with test data  \n  \nTN: {tn}  \nFP: {fp}  \nFN: {fn}  \nTP: {tp}'
    writer.add_text('Confusion Matrices/DT with Test data', data_summary)
    writer.add_text('Accuracy/Accuracy of DT', f'Accuracy with DT before reg: {acc_DT:.4f}')
    writer.add_figure(f'Decision Trees/DT before regularisation, Accuracy: {acc_DT:.4f}', fig_DT)
    plt.close(fig_DT)

    fig = plt.figure()
    plt.scatter(*X_train.T, c=colormap(y_train), edgecolors='k')
    plt.xlim([space[0][0], space[0][1]])
    plt.ylim([space[1][0], space[1][1]])
    plt.title('Training data')

    x_decision_fun = np.linspace(space[0][0], space[0][1], 100)
    y_decision_fun = fun(x_decision_fun)

    plt.plot(x_decision_fun, y_decision_fun, 'k-')
    plt.savefig(f'{path}/samples_training_plot.png')

    #plt.show()
    writer.add_figure('Training samples', figure=fig)
    plt.close(fig)
    data_summary = f'Samples: {num_samples}  \nTraining data shape: {X_train.shape}  \nTest data shape: {X_test.shape}'
    writer.add_text('Training Data Summary', data_summary)

    # Data preparation (to Tensor then create DataLoader for batch training)
    data_train_loader, data_test_loader = get_data_loader(X_train, y_train, X_test, y_test, args.batch)

    ############# Training ######################
    print('================Training===================')
    model, criterion, device = train(data_train_loader, data_test_loader, writer, ccp_alpha, regulariser, strength, dim, path, args)

    ############# Evaluation ######################
    print('================Test=======================')
    model.eval()
    y_train_predicted = []
    y_test_predicted = []
    loss_with_train_data = []
    loss_with_test_data = []
    with torch.no_grad():
        # Test with training data
        for i, batch in enumerate(data_train_loader):
            x, y = batch[0].to(device), batch[1].to(device)
            y_hat = model(x)
            y_train_predicted.append(y_hat)
            loss = criterion(input=y_hat, target=y)
            loss_with_train_data.append(loss.item())

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

        xx, yy = np.linspace(space[0][0], space[0][1], 100), np.linspace(space[1][0], space[1][1], 100)
        xx, yy = np.meshgrid(xx, yy)
        Z = pred_contours(xx, yy, model)
        Z = Z.reshape(xx.shape)

        fig = plt.figure()
        plt.scatter(*X_train.T, c=colormap(y_train), edgecolors='k')
        plt.title('Training data: ground truth')
        plt.xlim([space[0][0], space[0][1]])
        plt.ylim([space[1][0], space[1][1]])
        plt.plot(x_decision_fun, y_decision_fun, 'k-')
        writer.add_figure(f'Inference/Training data ground truth',
                          figure=fig)
        plt.close(fig)

        fig = plt.figure()
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
        CS = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
        plt.colorbar()
        plt.contour(xx, yy, Z, CS.levels, colors='k', linewidths=1.5)

        plt.scatter(*X_train.T, c=colormap(y_train_predicted), edgecolors='k')
        plt.xlim([space[0][0], space[0][1]])
        plt.ylim([space[1][0], space[1][1]])
        plt.title('Training data prediction')

        plt.plot(x_decision_fun, y_decision_fun, 'k-')

        fig.tight_layout()

        writer.add_figure(f'Inference/Inference with training data, loss: {np.array(loss_with_train_data).mean()}',
                          figure=fig)

        plt.close(fig)

        fig = plt.figure()
        plt.scatter(*X_test.T, c=colormap(y_test), edgecolors='k')
        plt.title('Test data: ground truth')
        plt.xlim([space[0][0], space[0][1]])
        plt.ylim([space[1][0], space[1][1]])
        writer.add_figure(f'Inference/Test data ground truth',
                          figure=fig)

        plt.close(fig)

        fig = plt.figure()
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
        CS = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
        plt.colorbar()
        plt.contour(xx, yy, Z, CS.levels, colors='k', linewidths=1.5)

        #plt.scatter(*X_train.T, c=colormap(y_train), edgecolors='k')
        plt.xlim([space[0][0], space[0][1]])
        plt.ylim([space[1][0], space[1][1]])
        plt.title('Network Contourplot with Training data')

        plt.savefig(f'{path}/fig_test_prediction.png')

        writer.add_figure(f'Inference/Inference with test data, loss: {np.array(loss_with_test_data).mean()}',
                          figure=fig)

        plt.close(fig)

        y_pred = [1 if y > 0.5 else 0 for y in y_train_predicted]
        tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
        acc_NN_train = accuracy_score(y_train, y_pred)
        data_summary = f'NN with train data  \n  \nTP: {tp}  \nFP: {fp}  \nFN: {fn}  \nTN: {tn}'
        writer.add_text('Confusion Matrices/NN with Train data', data_summary)
        writer.add_text('Accuracy/Accuracy of NN with Train data', f'Accuracy of NN with train data: {acc_NN_train:.4f}')

        y_pred = [1 if y > 0.5 else 0 for y in y_test_predicted]
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        acc_NN_test = accuracy_score(y_test, y_pred)
        data_summary = f'NN with test data  \n  \nTP: {tp}  \nFP: {fp}  \nFN: {fn}  \nTN: {tn}'
        writer.add_text('Confusion Matrices/NN with Test data', data_summary)
        writer.add_text('Accuracy/Accuracy of NN with Test data', f'Accuracy of NN with test data: {acc_NN_test:.4f}')

    # Decision tree after regularization

    fig_DT_reg, fig_contour, y_hat_tree, ccp_alpha = build_decision_tree(X_train, y_train_predicted, X_test, y_test, space, f"{path}/decision_tree_reg", ccp_alpha=ccp_alpha)
    acc_DT_reg = accuracy_score(y_test, y_hat_tree)
    tn, fp, fn, tp = confusion_matrix(y_test, y_hat_tree).ravel()
    data_summary = f'DT before reg with test data  \n  \nTN: {tn}  \nFP: {fp}  \nFN: {fn}  \nTP: {tp}'
    writer.add_text('Confusion Matrices/DT with Test data', data_summary)
    writer.add_text('Accuracy/Accuracy of DT', f'Accuracy with DT after reg: {acc_DT_reg:.4f}')
    writer.add_figure(f'Decision Trees/DT after regularisation, Accuracy: {acc_DT_reg:.4f}', fig_DT_reg)
    plt.close(fig_DT_reg)

    # Final outputs

    print(f'Accuracy of NN with training data: {acc_NN_train:.4f}')
    print(f'Accuracy of NN with test data: {acc_NN_test:.4f}')
    print(f'Accuracy of NN DT before regularisation with test data: {acc_DT:.4f}')
    print(f'Accuracy of NN DT after regularisation with test data: {acc_DT_reg:.4f}')

    writer.close()
    del model


if __name__ == '__main__':

    args = parser().parse_args()
    strengths = [0.1, 0.5, 1, 5, 10, 25, 75, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 25000, 50000, 75000, 100000]
    temp_strength = 200

    fig_dir = 'figures/test'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    init(fig_dir, temp_strength, 'tr')

    # for i, s in enumerate(strengths):
    #     os.makedirs(f'figures/{i}')
    #     init(i, s)

    # loss = []
    #
    # rows = 3
    # cols = 5
    # fig, ax = plt.subplots(rows, cols)
    # total = len(strengths)
    # counter = 1
    #
    #
    # for i in range(0, total, 5):
    #     for i in range(i, i+5):
    #         training_loss = main(strength=strengths[i])
    #         img = ImagePIL.open(f'figures/fig_test_prediction.png')
    #         ax[i][0].imshow(img)
    #         ax[i][0].set_title(f'Test Pred, $\lambda$: {strengths[i]}')
    #
    #
    #         img = ImagePIL.open(f'figures/decision_tree_reg.png')
    #         ax[i][1].imshow(img)
    #         ax[i][1].set_title(f'DT, $\lambda$: {strengths[i]}')
    #
    #
    #         img = ImagePIL.open(f'figures/decision_tree.png')
    #         ax[i][2].imshow(img)
    #         ax[i][2].set_title(f'DT_reg, $\lambda$: {strengths[i]}')
    #
    #
    #         ax[i][3].plot(range(0, len(training_loss)), training_loss)
    #         ax[i][3].set_title(f'Loss, $\lambda$: {strengths[i]}')
    #
    #
    #
    #     fig.savefig(f'figures/contour_plots_{counter}.png')
    #     counter += 1
    #     #plt.show()
    #     plt.close(fig)
