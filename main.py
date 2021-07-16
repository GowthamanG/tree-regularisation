import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from datasets import sample_2D_data, parabola
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import networks
import tree_regularisation as tr
import decision_tree_utils as dtu
from utils import save_data, get_data_loader, colormap, build_decision_tree, augment_data, pred_contours
from sklearn.metrics import accuracy_score
import argparse
from PIL import Image as ImagePIL


def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path',
                        type=str,
                        required=False,
                        default='figures',
                        help='Path wherein the figures should be stored')

    parser.add_argument('--ep',
                        type=int,
                        required=False,
                        default=250,
                        help='Number of epochs, default 250')

    parser.add_argument('--batch',
                        type=int,
                        default=100,
                        required=False,
                        help='Batch size, default 100')

    parser.add_argument('--lr',
                        type=float,
                        required=False,
                        default=1e-3,
                        help='Learning rate, default 1e-3')

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
                        default=1e-3,
                        required=False,
                        help='Regularization strength for the surrogate training, default 1e-3')

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
                        default=25,
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

    surrogate_model, optimizer_state_dict, sr_loss, test_loss = tr.train_surrogate_model(
                                                    params=X,
                                                    APLs=y,
                                                    epsilon=epsilon,
                                                    learning_rate=learning_rate,
                                                    current_optimizer=optimizer,
                                                    current_surrogate_model=model)

    return surrogate_model, optimizer_state_dict, sr_loss, test_loss


def train_surrogate_model_with_aggregation(W, APLs, learning_rate, epsilon, optimizer=None, model=None):
    X = torch.vstack(W)
    y = torch.tensor([APLs], dtype=torch.float).T

    surrogate_model, optimizer_state_dict, sr_loss, test_loss = tr.train_surrogate_model(
                                                    params=X,
                                                    APLs=y,
                                                    epsilon=epsilon,
                                                    learning_rate=learning_rate,
                                                    current_optimizer=optimizer,
                                                    current_surrogate_model=model)

    return surrogate_model, optimizer_state_dict, sr_loss, test_loss


def train(data_train_loader, data_test_loader, writer, ccp_alpha, regulariser, strength, dim, path, args):

    model = networks.Net2(input_dim=dim)
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
    #criterion = TreeRegularisedLoss(nn.BCEWithLogitsLoss(), regularization_strength)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    surrogate_model = None
    optimizer_surrogate_model = None
    surrogate_model_trained = False

    num_iter = args.sr_batch
    input_data_st = []
    APLs = []
    training_loss = []
    training_loss_without_reg = []
    validation_loss = []

    surrogate_training_loss = []
    surrogate_validation_loss = []

    omega_plot = []
    sr_val_loss = []

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
                    surrogate_model, optimizer_surrogate_model, train_loss, val_loss = train_surrogate_model_with_aggregation(
                        input_data_st + input_data_st_augmented,
                        APLs + APLs_augmented,
                        args.lr_sr,
                        args.epsilon,
                        optimizer=optimizer_surrogate_model,
                        model=surrogate_model)


                else:
                    surrogate_model, optimizer_surrogate_model, train_loss, val_loss = train_surrogate_model_with_aggregation(
                        input_data_st + input_data_st_augmented,
                        APLs + APLs_augmented,
                        args.lr_sr,
                        args.epsilon)

            else:
                # Train surrogate model without input (weights) aggregation
                if args.sw:
                    surrogate_model, optimizer_surrogate_model, train_loss, val_loss = train_surrogate_model(
                        input_data_st + input_data_st_augmented,
                        APLs + APLs_augmented,
                        args.lr_sr,
                        args.epsilon,
                        optimizer=optimizer_surrogate_model,
                        model=surrogate_model)
                else:
                    surrogate_model, optimizer_surrogate_model, train_loss, val_loss = train_surrogate_model(
                        input_data_st + input_data_st_augmented,
                        APLs + APLs_augmented,
                        args.lr_sr,
                        args.epsilon)

                input_data_st = []
                APLs = []

            num_iter = args.sr_batch
            surrogate_model_trained = True
            surrogate_training_loss.append(train_loss)
            surrogate_validation_loss.append(val_loss)

        # Training loop of the first network
        for i, batch in enumerate(data_train_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)

            y_hat = model(x_batch)

            if surrogate_model_trained:

                for param in surrogate_model.parameters():
                    param.data.requires_grad = False

                #surrogate_model.eval()

                regularisation_term  = lambda x: {
                    'l1': torch.norm(x, 1),
                    'l2': torch.norm(x, 2),
                    'tr': surrogate_model(x)
                }

                omega = regularisation_term(model.parameters_to_vector())[regulariser]

                loss = 0*criterion(input=y_hat, target=y_batch) + regularization_strength * omega
                loss_without_reg = criterion(input=y_hat, target=y_batch) # Only for plotting purpose

                batch_loss_without_reg.append(loss_without_reg.item())
                omega_plot.append(regularisation_term(model.parameters_to_vector())[regulariser])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            else:
                loss = criterion(input=y_hat, target=y_batch)
                batch_loss_without_reg.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_loss.append(loss.item())

            # Stack model parameters and APLs after every epoch for surrogate training
            average_path_length = dtu.average_path_length(X_train=data_train_loader.dataset[:][0].to(device),
                                                          X_test=data_test_loader.dataset[:][0],
                                                          y_test=data_test_loader.dataset[:][1],
                                                          model=model,
                                                          ccp_alpha=ccp_alpha)

            # Collect weights and APLs for surrogate training
            input_data_st.append(model.parameters_to_vector())
            APLs.append(average_path_length)

        print(f'Epoch: [{epoch + 1}/{num_epochs}, Loss: {np.array(batch_loss).mean():.4f}]')
        training_loss.append(np.array(batch_loss).mean())
        training_loss_without_reg.append(np.array(batch_loss_without_reg).mean())

        num_iter -= 1

    surrogate_training_loss = torch.tensor(surrogate_training_loss).flatten()
    surrogate_validation_loss = torch.tensor(surrogate_validation_loss).flatten()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.plot(range(0, len(training_loss)), training_loss)
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ax1.grid()
    ax1.set_title(f'Training loss, $\lambda$: {regularization_strength}, {regulariser}')

    ax2.plot(range(0, len(surrogate_training_loss)), surrogate_training_loss)
    ax2.set_xlabel('training iteration')
    ax2.set_ylabel('loss')
    ax2.grid()
    ax2.set_title(f'Surrogate Training Loss, $\lambda$: {regularization_strength}, {regulariser}')

    ax3.plot(range(0, len(surrogate_validation_loss)), surrogate_validation_loss, c='r')
    ax3.set_xlabel('training iteration')
    ax3.set_ylabel('loss')
    ax3.grid()
    ax3.set_title(f'Surrogate Validation Loss, $\lambda$: {regularization_strength}, {regulariser}')

    fig.tight_layout()
    fig.savefig(f'{path}/loss.png')
    plt.close(fig)

    for i, value in enumerate(training_loss):
        writer.add_scalar('Training Loss', value, i)

    for i, value in enumerate(training_loss_without_reg):
        writer.add_scalar(f'Training Loss without Regularisation', value, i)

    for i, value in enumerate(omega_plot):
        writer.add_scalar(f'Omega Values: {regulariser}', value, i)

    for i, (train_loss, val_loss) in enumerate(zip(surrogate_training_loss, surrogate_validation_loss)):
        writer.add_scalars(f'Surrogate Training Loss', {
            'train loss': train_loss,
            'validation loss': val_loss
        }, i)

    del input_data_st
    del APLs
    del surrogate_model

    return model, criterion, device


def init(path, strength, regulariser):

    num_samples, dim, space = 2000, 2, [[0, 1.5], [0, 1.5]]
    writer = SummaryWriter(log_dir=f'runs/{regulariser}_25_{strength}')

    fun = parabola # either use paraobla, polynom_3, polynom_3 or create a new one
    if args.save:
        X, Y = sample_2D_data(num_samples, fun, space)
        save_data(X, Y, 'feed_forward_network/dataset/parabola/data_parabola')

    data_from_txt = np.loadtxt('dataset/parabola/data_parabola.txt')
    train_data_from_txt = np.loadtxt('dataset/parabola/data_parabola_train.txt')
    test_data_from_txt = np.loadtxt('dataset/parabola/data_parabola_test.txt')

    X, Y = data_from_txt[:, :2], data_from_txt[:, 2]
    X_train, y_train = train_data_from_txt[:, :2], train_data_from_txt[:, 2]
    X_test, y_test = test_data_from_txt[:, :2], test_data_from_txt[:, 2]

    # Decision tree directly on input space
    fig_DT, fig_contour, y_hat_tree, ccp_alpha = build_decision_tree(X, Y, X_train, y_train, X_test, y_test, space, f"{path}/decision_tree")
    acc_DT = accuracy_score(y_test, y_hat_tree)
    #tn, fp, fn, tp = confusion_matrix(y_test, y_hat_tree).ravel()

    #data_summary = f'DT before reg with test data  \n  \nTN: {tn}  \nFP: {fp}  \nFN: {fn}  \nTP: {tp}'
    #writer.add_text('Confusion Matrices/DT with Test data', data_summary)
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
    X_train_temp = [] # Because training data are shuffled, collect them for plotting afterwards
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
        #plt.colorbar()
        #plt.contour(xx, yy, Z, CS.levels, colors='k', linewidths=1.5)
        plt.scatter(*X_train_temp.T, c=colormap(y_train_predicted), edgecolors='k')
        plt.xlim([space[0][0], space[0][1]])
        plt.ylim([space[1][0], space[1][1]])
        plt.title('Network Contourplot with Training data')
        #plt.plot(x_decision_fun, y_decision_fun, 'k-')
        fig.tight_layout()
        plt.savefig(f'{path}/fig_train_prediction.png')
        writer.add_figure(f'Inference/Inference with training data, loss: {np.array(loss_with_train_data).mean()}', figure=fig)
        plt.close(fig)

        # Scatterplot with test data
        fig = plt.figure()
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
        CS = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
        plt.scatter(*data_test_loader.dataset[:][0].T, c=colormap(y_test_predicted), edgecolors='k')
        plt.title('Network Contourplot with Test data')
        plt.xlim([space[0][0], space[0][1]])
        plt.ylim([space[1][0], space[1][1]])
        plt.savefig(f'{path}/fig_test_prediction.png')
        writer.add_figure(f'Inference/Inference with test data, loss {np.array(loss_with_test_data).mean()}', figure=fig)
        plt.close(fig)

        y_train_predicted = [1 if y > 0.5 else 0 for y in y_train_predicted]
        tn, fp, fn, tp = confusion_matrix(y_train_temp, y_train_predicted).ravel()
        acc_NN_train = accuracy_score(y_train_temp, y_train_predicted)
        data_summary = f'NN with train data  \n  \nTP: {tp}  \nFP: {fp}  \nFN: {fn}  \nTN: {tn}'
        writer.add_text('Confusion Matrices/NN with Train data', data_summary)
        writer.add_text('Accuracy/Accuracy of NN with Train data', f'Accuracy of NN with train data: {acc_NN_train:.4f}')

        y_test_predicted = [1 if y > 0.5 else 0 for y in y_test_predicted]
        tn, fp, fn, tp = confusion_matrix(y_test, y_test_predicted).ravel()
        acc_NN_test = accuracy_score(y_test, y_test_predicted)
        data_summary = f'NN with test data  \n  \nTP: {tp}  \nFP: {fp}  \nFN: {fn}  \nTN: {tn}'
        writer.add_text('Confusion Matrices/NN with Test data', data_summary)
        writer.add_text('Accuracy/Accuracy of NN with Test data', f'Accuracy of NN with test data: {acc_NN_test:.4f}')

    # Decision tree after regularization

    fig_DT_reg, fig_contour, y_hat_tree, ccp_alpha = build_decision_tree(X, Y, X_train_temp, y_train_predicted, X_test, y_test, space, f"{path}/decision_tree_reg", ccp_alpha)
    acc_DT_reg = accuracy_score(y_test, y_hat_tree)
    #tn, fp, fn, tp = confusion_matrix(y_test, y_hat_tree).ravel()

    #data_summary = f'DT before reg with test data  \n  \nTN: {tn}  \nFP: {fp}  \nFN: {fn}  \nTP: {tp}'
    #writer.add_text('Confusion Matrices/DT with Test data', data_summary)
    writer.add_text('Accuracy/Accuracy of DT', f'Accuracy with DT after reg: {acc_DT_reg:.4f}')
    writer.add_figure(f'Decision Trees/DT after regularisation, Accuracy: {acc_DT_reg:.4f}', fig_DT_reg)
    writer.add_figure(f'Decision Trees/DT Contourplot after regularisation, Accuracy: {acc_DT_reg:.4f}', fig_contour)
    plt.close(fig_DT_reg)

    dt = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
    dt.fit(X_train_temp, y_train_predicted)
    plot_confusion_matrix(dt, X_test, y_test)
    plt.title("Confusion Matrix Tree regularized NN")
    plt.savefig(f'{path}/confusion_matrix_regularized_tree.png')
    img = ImagePIL.open(f'{path}/confusion_matrix_regularized_tree.png')
    fig = plt.figure()
    plt.imshow(img)

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
    regulariser = 'tr'
    strength = 1

    fig_path = f'figures/{regulariser}_25_{strength}'
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    init(fig_path, strength, regulariser)
