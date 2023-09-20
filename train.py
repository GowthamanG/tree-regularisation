from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from datasets import parabola, cos, sample_2D_data
import networks
from utils import *
import argparse
from PIL import Image as ImagePIL

np.random.seed(5555)
torch.random.manual_seed(5255)


def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--label',
                        required=False,
                        type=str,
                        default='no_label',
                        help='Additional label as postfix to the directory path name to indicate this run')


    parser.add_argument('--lambda_init',
                        required=False,
                        type=float,
                        default=1e-3,
                        help='Initial lambda value as regularisation term')

    parser.add_argument('--lambda_target',
                        required=False,
                        type=float,
                        default=1,
                        help='Target lambda value as regularisation term')

    parser.add_argument('--ep',
                        required=False,
                        default=1000,
                        type=int,
                        help='Total number of epochs, default 1000 (300 warm up + 700 regularisation)')

    parser.add_argument('--min_samples_leaf',
                        required=False,
                        default=5,
                        type=int,
                        help='Minimum samples leaf for pre-pruning, default 5')

    parser.add_argument('--batch',
                        default=1024,
                        required=False,
                        help='Batch size, default 1024')

    return parser


def resample_data():

    if fun_name == "parabola":
        X, y = sample_2D_data(5000, parabola, 0.2, space)
    elif fun_name == "cos":
        X, y = sample_2D_data(5000, cos, 0.4, space)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=50)

    X_train = torch.tensor(X_train, dtype=torch.float).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float).to(device)
    y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float).to(device)
    y_test = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float).to(device)

    data_train = TensorDataset(X_train, y_train)
    data_train_loader = DataLoader(dataset=data_train, batch_size=64, shuffle=True)
    data_test = TensorDataset(X_test, y_test)
    data_test_loader = DataLoader(dataset=data_test, batch_size=64)

    return data_train_loader, data_test_loader


def model_contour_plot(space, model, plot_title, fig_file_name, X=None, y=None):
    """
    Draw contour plot for deep model.

    Parameters
    -------

    space: Feature space

    model: Target deep model

    plot_title: Plot title

    fig_file_name: Data name for saving the figure

    X: Input features, default None

    y: Labels, default None
    """

    xx, yy = np.linspace(space[0][0], space[0][1], 100), np.linspace(space[1][0], space[1][1], 100)
    xx, yy = np.meshgrid(xx, yy)
    Z = pred_contours(xx, yy, model).reshape(xx.shape)

    fig = plt.figure()
    CS = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
    #plt.colorbar()
    # plt.contour(xx, yy, Z, CS.levels, colors='k', linewidths=1.5)
    if X is not None:
        plt.scatter(*X.T, c=colormap(y), edgecolors='k')
    plt.xlim([space[0][0], space[0][1]])
    plt.ylim([space[1][0], space[1][1]])
    plt.title(plot_title)
    #fig.tight_layout()
    plt.savefig(fig_file_name)
    plt.close(fig)


def snap_shot_train(data_test_loader, criterion, lambda_, model, accuracy, epoch, path):
    """
    Making snapshot during the training process. Save deep model's contourplot, the associated decision tree and its
    contour plot.

    Parameters
    -------

    data_test_loader: Input data loader for test data

    criterion: Deep model's loss function

    model: Target deep model

    accuracy: Current accuracy measure of the deep model

    epoch: Current training epoch

    path: Directory path, where the plots should be saved

    """
    y_train_predicted = []
    X_train_temp = []
    y_train_temp = []

    data_train_loader_, data_test_loader_ = resample_data()

    with torch.no_grad():
        # Test with training data
        for i, batch in enumerate(data_train_loader_):
            x, y = batch[0].to(device), batch[1].to(device)
            X_train_temp.append(x)
            y_train_temp.append(y)

            y_hat = model(x)
            y_train_predicted.append(y_hat)
            loss = criterion(input=y_hat, target=y)

        X_train_temp = torch.cat(X_train_temp).cpu().numpy()
        y_train_temp = torch.cat(y_train_temp).cpu().numpy()
        y_train_predicted = torch.cat(y_train_predicted)
        y_train_predicted = torch.where(y_train_predicted > 0.5, 1, 0).detach().cpu().float().numpy().reshape(-1)

    X_test_, y_test_ = dataloader_to_numpy(data_test_loader_)

    _ = build_decision_tree(X_train_temp, y_train_predicted, X_test_, y_test_, space,
                            f"{path}/decision_tree-snapshot-epoch-{epoch}", epoch=epoch)

    plot_title = f'Network Contourplot, $\lambda$: {lambda_}, Accuracy: {accuracy:.2f}'
    fig_file_name = f'{path}/fig_train_prediction-snapshot-epoch-{epoch}.png'
    model_contour_plot(space, model, plot_title, fig_file_name)


def train_surrogate_model(X, y, criterion, optimizer, model):

    X_train = torch.vstack(X).detach()
    y_train = torch.tensor([y], dtype=torch.float).T.to(device)

    model.surrogate_network.to(device)

    num_epochs = 5
    batch_size = 256

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


def train(data_train_loader, data_test_loader, data_val_loader, writer, path):

    model = networks.TreeNet(input_dim=dim, min_samples_leaf=args.min_samples_leaf)
    model.to(device)

    # Hypterparameters
    num_random_restarts = 50
    total_num_epochs = args.ep
    epochs_warm_up = 300
    epochs_reg = total_num_epochs - epochs_warm_up
    lambda_init = args.lambda_init
    lambda_target = args.lambda_target
    lambda_ = lambda_init

    alphas = {
        '0.5': -20,
        '1.0': -13000,
        '2.0': 20,
        '3.0': 13,
        '4.0': 10,
        '5.0': 8
    }

    alpha = alphas[str(float(lambda_target))]
    cooling_fun = lambda k: lambda_target + (lambda_init - lambda_target) * (1 / (1 + np.exp(((alpha * np.log((np.abs(lambda_init - lambda_target))) / epochs_reg) * (k - epochs_reg / 2)))))

    # Objectives and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.feed_forward.parameters(), lr=1e-4)

    criterion_sr = nn.MSELoss()
    optimizer_sr = Adam(model.surrogate_network.parameters(), lr=1e-3, weight_decay=1e-5)

    input_surrogate = []
    APLs_surrogate = []

    APLs_truth = []
    APL_predictions = []

    training_loss_without_reg = []
    val_loss = []
    training_accuracy = []
    tree_accuracy = []
    surrogate_training_loss = []

    lambdas = [lambda_]

    x_iter_warm_up = 0
    iters_per_epoch = 0

    for i in range(num_random_restarts):
        data_train_loader_new, _ = resample_data()

        model.reset_outer_weights()
        input_surrogate.append(model.parameters_to_vector())
        APL = model.compute_APL(data_train_loader_new.dataset[:][0])
        APLs_surrogate.append(APL)
        print(f'Random restart [{i + 1}/{num_random_restarts}]')

    for epoch in range(total_num_epochs):
        model.train()
        batch_loss_val = []
        batch_loss_without_reg = []
        batch_accuracy = []

        if epoch > 0:

            if epoch > (epochs_warm_up - 1):
                lambda_ = cooling_fun(epoch - epochs_warm_up)
                lambdas.append(lambda_)

            input_surrogate_augmented, APLs_surrogate_augmented = augment_data_with_dirichlet(data_train_loader.dataset[:][0], input_surrogate, networks.TreeNet(input_dim=dim), device, 500)
            model.freeze_model()
            model.surrogate_network.unfreeze_model()

            input_surrogate_augmented = input_surrogate + input_surrogate_augmented
            APLs_surrogate_augmented = APLs_surrogate + APLs_surrogate_augmented
            sr_train_loss = train_surrogate_model(input_surrogate_augmented, APLs_surrogate_augmented, criterion_sr, optimizer_sr, model)
            surrogate_training_loss.append(sr_train_loss)

            model.surrogate_network.freeze_model()
            model.unfreeze_model()
            model.surrogate_network.eval()

            del input_surrogate_augmented
            del APLs_surrogate_augmented
            del sr_train_loss

        for (x, y) in data_train_loader:

            y_hat = model(x)

            if epoch > (epochs_warm_up - 1): # regularisation phase
                omega = model.compute_APL_prediction()
                loss = criterion(input=y_hat, target=y) + lambda_ * omega
            else: # warm-up phase
                loss = criterion(input=y_hat, target=y)
                x_iter_warm_up += 1

            loss_without_reg = criterion(input=y_hat, target=y)  # Only for plotting, not for optimisation
            batch_loss_without_reg.append(float(loss_without_reg))
            del loss_without_reg

            APL_predictions.append(model.compute_APL_prediction())

            iters_per_epoch += 1 if epoch == 0 else 0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Collect weights and APLs for surrogate training
            input_surrogate.append(model.parameters_to_vector())
            data_train_loader_new, _ = resample_data()
            APL = model.compute_APL(data_train_loader_new.dataset[:][0])
            APLs_surrogate.append(APL)
            APLs_truth.append(APL)

            del x, y

        if epoch > 0 and epoch % 10 == 0:  # snapshots of the resulting tree
            torch.save(model.state_dict(), f'models/model_snapshot_{epoch}.pth')
            model.eval()
            model.freeze_model()
            snap_shot_train(data_test_loader, criterion, lambda_, model, tree_accuracy[-1], epoch, path)
            model.unfreeze_model()
            model.train()

        # Validation
        model.eval()
        for (x, y) in data_val_loader:
            y_hat = model(x)
            loss = criterion(input=y_hat, target=y)
            batch_loss_val.append(float(loss))
            y_hat = torch.where(y_hat > 0.5, 1, 0).cpu().numpy()
            y = y.detach().cpu().numpy()
            batch_accuracy.append(accuracy_score(y, y_hat))

            del x, y

        print(f'Epoch: [{epoch + 1}/{total_num_epochs}, Loss: {np.array(batch_loss_without_reg).mean():.4f}]')

        training_loss_without_reg.append(np.array(batch_loss_without_reg).mean())
        val_loss.append(np.array(batch_loss_val).mean())
        training_accuracy.append(np.array(batch_accuracy).mean())

        data_train_loader_new, data_test_loader_new = resample_data()
        X_train_new, y_train_new = dataloader_to_numpy(data_train_loader_new)
        X_test_new, y_test_new = dataloader_to_numpy(data_test_loader_new)

        ccp_alpha = post_pruning(X_train_new, y_train_new)
        dt = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha, min_samples_leaf=args.min_samples_leaf)
        y_hat_ = model(data_train_loader_new.dataset[:][0])
        y_hat_ = torch.where(y_hat_ > 0.5, 1, 0).detach().cpu().numpy()
        dt.fit(X_train_new, y_hat_)
        acc = accuracy_score(y_test_new, dt.predict(X_test_new))
        tree_accuracy.append(acc)

    # PLOTS
    surrogate_training_loss = torch.tensor(surrogate_training_loss).flatten()

    fig = plt.figure()
    plt.plot(range(0, len(training_loss_without_reg)), training_loss_without_reg, label='Training loss')
    plt.plot(range(0, len(val_loss)), val_loss, label='Validation loss')
    plt.xlabel(f'epochs ({iters_per_epoch} iterations per epoch)')
    plt.ylabel('loss')
    plt.legend()
    plt.grid()
    plt.title('Training loss')
    fig.tight_layout()
    fig.savefig(f'{path}/training_loss.png')
    plt.close(fig)

    fig = plt.figure()
    plt.plot(range(0, len(surrogate_training_loss)), surrogate_training_loss)
    plt.xlabel(f'epochs ({iters_per_epoch} iterations per epoch)')
    plt.ylim([0, 1.0])
    plt.ylabel('loss')
    plt.grid()
    plt.title(f'Surrogate Training Loss')
    fig.tight_layout()
    fig.savefig(f'{path}/surrogate_training_loss.png')
    plt.close(fig)

    fig = plt.figure()
    plt.plot(range(0, len(APLs_truth)), APLs_truth, color='y', label='true APL')
    plt.plot(range(0, len(APL_predictions)), APL_predictions, color='g', label='predicted APL $\hat{\Omega}(W)$')
    plt.vlines(x_iter_warm_up, 0, max(APLs_truth), linestyles="dashed", colors='r')
    plt.xlabel('iterations')
    plt.ylabel('path length')
    plt.legend()
    plt.annotate("warm up", (x_iter_warm_up/2, 0.5))
    plt.annotate("regularization", (x_iter_warm_up + 1000, 0.5))
    plt.grid()
    plt.title(f'Path length estimates')
    fig.tight_layout()
    fig.savefig(f'{path}/path_estimates.png')
    plt.close(fig)

    fig = plt.figure()
    plt.plot(range(0, len(tree_accuracy)), tree_accuracy, color='b', label='Accuracy Decision Trees')
    plt.plot(range(0, len(training_accuracy)), training_accuracy, color='r', label='Accuracy Network')
    plt.xlabel(f'epochs ({iters_per_epoch} iterations per epoch)')
    plt.ylabel('accuracy')
    plt.legend()
    plt.grid()
    plt.title(f'Accuracy')
    fig.tight_layout()
    fig.savefig(f'{path}/accuracy.png')
    plt.close(fig)

    fig = plt.figure()
    plt.plot(range(epochs_warm_up, len(lambdas) + epochs_warm_up), lambdas)
    plt.xlabel(f'epochs ({iters_per_epoch} iterations per epoch)')
    plt.ylabel('$\lambda$')
    plt.grid()
    plt.title(f'$\lambda$ curve')
    plt.savefig(f'{path}/lambda_curve.png')
    plt.close(fig)

    for i, _ in enumerate(surrogate_training_loss):
        for j, value in enumerate(surrogate_training_loss[i]):
            writer.add_scalar(f'Surrogate Training/Loss of surrogate training after epoch {i}', value, j)

    for i, value in enumerate(training_loss_without_reg):
        writer.add_scalar(f'Training loss without regularisation', value, i)

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

    writer = SummaryWriter(log_dir=tb_logs_path)

    train_data_from_txt = np.loadtxt(f'dataset/{fun_name}/data_{fun_name}_train.txt')
    test_data_from_txt = np.loadtxt(f'dataset/{fun_name}/data_{fun_name}_test.txt')
    val_data_from_txt = np.loadtxt(f'dataset/{fun_name}/data_{fun_name}_val.txt')

    # Data preparation, first to Tensor then create DataLoader to get mini-batches
    X_train, y_train = train_data_from_txt[:, :2], train_data_from_txt[:, 2]
    X_test, y_test = test_data_from_txt[:, :2], test_data_from_txt[:, 2]
    X_val, y_val = val_data_from_txt[:, :2], val_data_from_txt[:, 2]
    data_train_loader, data_test_loader, data_val_loader = get_data_loader(X_train, y_train, X_test, y_test, X_val,
                                                                           y_val, torch.float, torch.float, args.batch)

    # Decision tree, where data is directly fed into
    tree_accuracy = build_decision_tree(X_train, y_train, X_test, y_test, space, f"{path}/decision_tree_original_data")

    x_decision_fun = np.linspace(space[0][0], space[0][1], 100)
    y_decision_fun = fun(x_decision_fun)

    fig = plt.figure()
    plt.scatter(*X_train.T, c=colormap(y_train), edgecolors='k')
    plt.xlim([space[0][0], space[0][1]])
    plt.ylim([space[1][0], space[1][1]])
    plt.title('Training data')
    plt.plot(x_decision_fun, y_decision_fun, 'k-', linewidth=2.5)
    plt.plot(x_decision_fun, y_decision_fun - 0.2, linewidth=2, color='#808080')
    plt.plot(x_decision_fun, y_decision_fun + 0.2, linewidth=2, color='#808080')
    plt.savefig(f'{path}/samples_training_plot.png')
    writer.add_figure('Training samples', figure=fig)
    plt.close(fig)
    data_summary = f'Training data shape: {X_train.shape}  \nValidation data shape: {X_val.shape}, \nTest data shape: {X_test.shape}'
    writer.add_text('Training data Summary', data_summary)

    ############# Training ######################
    print('Training'.center(len('Training') + 2).center(30, '='))
    model, criterion = train(data_train_loader, data_test_loader, data_val_loader, writer, path)

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
        for (x, y) in data_train_loader:
            X_train_temp.append(x)
            y_train_temp.append(y)

            y_hat = model(x)
            y_train_predicted.append(y_hat)
            loss = criterion(input=y_hat, target=y)
            loss_with_train_data.append(loss.item())

        X_train_temp = torch.cat(X_train_temp).cpu().numpy()
        y_train_temp = torch.cat(y_train_temp).cpu().numpy()
        y_train_predicted = torch.cat(y_train_predicted)
        y_train_predicted = torch.where(y_train_predicted > 0.5, 1, 0).detach().cpu().numpy()

        ## PLOTS ##
        plot_title = 'Model Contourplot with Training data'
        fig_file_name = f'{path}/fig_train_prediction.png'
        model_contour_plot(space, model, plot_title, fig_file_name, X=X_train_temp, y=y_train_predicted)

        # Test with test data
        for (x, y) in data_test_loader:
            y_hat = model(x)
            y_test_predicted.append(y_hat)
            loss = criterion(input=y_hat, target=y)
            loss_with_test_data.append(loss.item())

        y_test_predicted = torch.cat(y_test_predicted)
        y_test_predicted = torch.where(y_test_predicted > 0.5, 1, 0).detach().cpu().numpy()

        plot_title = 'Model Contourplot with Testdata'
        fig_file_name = f'{path}/fig_test_prediction.png'
        model_contour_plot(space, model, plot_title, fig_file_name, X=X_test, y=y_test_predicted)

        accuracy_network_train_data = accuracy_score(y_train_temp, y_train_predicted)
        writer.add_text('Accuracy/Accuracy of network with train data', f'Accuracy of network with train data: {accuracy_network_train_data:.4f}')

        accuracy_network_test_data = accuracy_score(y_test, y_test_predicted)
        writer.add_text('Accuracy/Accuracy of network with test data', f'Accuracy of network with test data: {accuracy_network_test_data:.4f}')

    # Decision tree after regularization
    data_train_loader_new, data_test_loader_new = resample_data()
    X_train_new, y_train_new = dataloader_to_numpy(data_train_loader_new)
    X_test_new, y_test_new = dataloader_to_numpy(data_test_loader_new)
    y_train_predicted_ = model(data_train_loader_new.dataset[:][0])
    y_train_predicted_ = torch.where(y_train_predicted_ > 0.5, 1, 0).detach().cpu().numpy().reshape(-1)
    tree_accuracy_reg = build_decision_tree(X_train_new, y_train_predicted_, X_test_new, y_test_new, space, f"{path}/decision_tree_reg", min_samples_leaf=args.min_samples_leaf)

    # Final outputs

    print(f'Accuracy of network with training data: {accuracy_network_train_data:.4f}')
    print(f'Accuracy of network with test data: {accuracy_network_test_data:.4f}')
    print(f'Accuracy of tree before regularisation with test data: {tree_accuracy:.4f}')
    print(f'Accuracy of tree after network regularisation with test data: {tree_accuracy_reg:.4f}')

    writer.close()
    del model

if __name__ == '__main__':

    device = "cuda:0" if torch.cuda.is_available() else 'cpu'

    args = parser().parse_args()
    dim = 2

    fun = parabola
    fun_name = 'parabola'
    space = [[0, 1.5], [0, 1.5]]

    #fun = cos
    #fun_name = 'cos'
    # space = [[-6, 6], [-2, 2]]

    dir_name = f'tree_reg_train_{args.lambda_init}_{args.lambda_target}_{args.label}'

    fig_path = f'figures/{dir_name}'
    tb_logs_path = f'runs/{dir_name}'

    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    if not os.path.exists(tb_logs_path):
        os.makedirs(tb_logs_path)

    init(fig_path, tb_logs_path)
