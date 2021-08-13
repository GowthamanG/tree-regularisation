import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import cross_val_score, GridSearchCV, ParameterGrid
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
from PIL import Image as ImagePIL
import pydotplus

np.random.seed(5555)
torch.random.manual_seed(5255)

def get_data_loader(X_train, y_train, X_test, y_test, X_val, y_val, batch_size):
    X_train = torch.tensor(X_train, dtype=torch.float)
    y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float)
    X_test = torch.tensor(X_test, dtype=torch.float)
    y_test = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float)
    X_val = torch.tensor(X_val, dtype=torch.float)
    y_val = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float)

    data_train = TensorDataset(X_train, y_train)
    data_train_loader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
    data_test = TensorDataset(X_test, y_test)
    data_test_loader = DataLoader(dataset=data_test, batch_size=batch_size)
    data_val = TensorDataset(X_val, y_val)
    data_val_loader = DataLoader(dataset=data_val, batch_size=batch_size)

    return data_train_loader, data_test_loader, data_val_loader


def save_data(X, Y, filename: str):
    file_data = open(filename + '.txt', 'w')
    file_train_data = open(filename + '_train.txt', 'w')
    file_test_data = open(filename + '_test.txt', 'w')
    file_val_data = open(filename + '_val.txt', 'w')

    # data split 70/15/15 ratio
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    np.savetxt(file_data, np.hstack((X, Y.reshape(-1, 1))))
    np.savetxt(file_train_data, np.hstack((X_train, y_train.reshape(-1, 1))))
    np.savetxt(file_test_data, np.hstack((X_test, y_test.reshape(-1, 1))))
    np.savetxt(file_val_data, np.hstack((X_val, y_val.reshape(-1, 1))))

    file_data.close()
    file_train_data.close()
    file_test_data.close()
    file_val_data.close()


def colormap(Y):
    return ['b' if y == 1 else 'r' for y in Y]


def post_pruning(X, y):
    # https://medium.com/swlh/post-pruning-decision-trees-using-python-b5d4bcda8e23
    # https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html#sphx-glr-auto-examples-tree-plot-cost-complexity-pruning-py

    clf = DecisionTreeClassifier(random_state=42)
    path = clf.cost_complexity_pruning_path(X, y)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    ccp_alphas = ccp_alphas[:-1]
    scores = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
        score = cross_val_score(clf, X, y, cv=3, scoring="neg_mean_squared_error", n_jobs=-1)
        scores.append(score)

    # average over folds, fix sign of mse
    fold_mse = -np.mean(scores, 1)
    # select the most parsimonous model (highest ccp_alpha) that has an error within one standard deviation of
    # the minimum mse.
    # I.e. the “one-standard-error” rule (see ESL or a lot of other tibshirani / hastie notes on regularization)
    #selected_alpha = np.max(ccp_alphas[fold_mse <= np.min(fold_mse) + np.std(fold_mse)])
    selected_alpha = ccp_alphas[np.argmax(fold_mse)]

    return selected_alpha

def post_pruning_2(X, y):
    # https://towardsdatascience.com/build-better-decision-trees-with-pruning-8f467e73b107

    full_tree = DecisionTreeClassifier(random_state=42)
    ccp_alphas = full_tree.cost_complexity_pruning_path(X, y)['ccp_alphas']

    ccp_alpha_grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        #scoring=make_scorer(accuracy_score),
        param_grid={'ccp_alpha': [alpha for alpha in ccp_alphas[:-1]]}
    )

    ccp_alpha_grid_search.fit(X, y)

    return ccp_alpha_grid_search.best_params_['ccp_alpha']


def build_decision_tree(X, y, X_train, y_train, X_test, space, path, ccp_alpha=None):

    if ccp_alpha:
        final_decision_tree = DecisionTreeClassifier(random_state=42)
        final_decision_tree.fit(X_train, y_train)
    else:
        ccp_alpha = post_pruning_2(X_train, y_train)
        final_decision_tree = DecisionTreeClassifier(random_state=42)
        final_decision_tree.fit(X_train, y_train)

    y_hat_with_tree = final_decision_tree.predict(X_test)

    dot_data = StringIO()
    export_graphviz(final_decision_tree, out_file=dot_data, filled=True, rounded=True, special_characters=True,
                    feature_names=['x', 'y'],
                    class_names=['0', '1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(f'{path}.png')
    Image(graph.create_png())
    img = ImagePIL.open(f'{path}.png')
    fig_DT = plt.figure()
    plt.imshow(img)
    plt.title(f'DT with alpha: {ccp_alpha}')
    plt.close(fig_DT)

    xx, yy = np.meshgrid(np.linspace(space[0][0], space[0][1], 100),
                         np.linspace(space[0][0], space[0][1], 100))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = final_decision_tree.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    fig_contour = plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
    #plt.scatter(*X.T, c=colormap(y), edgecolors='k')
    plt.scatter(*X_train.T, c=colormap(y_train), edgecolors='k')
    plt.title(f'DT Contourplot with alpha: {ccp_alpha}')
    plt.savefig(f'{path}_contourplot.png')
    plt.close(fig_contour)

    return fig_DT, fig_contour, y_hat_with_tree, ccp_alpha

def pred_contours(x, y, model):
    data = np.c_[x.ravel(), y.ravel()]
    y_pred = []

    for d in data:
        y_hat = model(torch.tensor(d, dtype=torch.float, device='cuda:0'))
        y_pred.append(y_hat.detach().cpu().numpy())

    y_pred = np.array(y_pred)
    y_pred = np.where(y_pred > 0.5, 1, 0)

    return y_pred


def augment_data_with_gaussian(X_train, model, device, size, ccp_alpha):

    parameters = []
    APLs = []

    for _ in range(size):

        model_copy = copy.deepcopy(model)
        model_copy.eval()

        for param in model_copy.feed_forward.parameters():
            param.data.requires_grad = False

            # variance: 0.1 - 0.3 times relative to the absolute value of the model parameter
            param_augmented = np.random.normal(param.data.cpu().numpy(), 0.1 * np.abs(param.data.cpu().numpy()))
            # param_augmented = np.random.normal(param.data.cpu().numpy(), 0.1)
            param.data = torch.tensor(param_augmented, dtype=torch.float).float().to(device)

        parameters.append(model_copy.get_parameter_vector)
        APLs.append(model_copy.compute_APL(X_train, ccp_alpha))

        del model_copy

    return parameters, APLs


def augment_data_with_dirichlet(X_train, parameters, model, device, num_new_samples, ccp_alpha):

    parameters_new = []
    APLs_new = []
    num_parameters = len(parameters)

    alpha = [1]*num_parameters
    samples = np.random.dirichlet(alpha, num_new_samples)
    parameters = torch.vstack(parameters)
    samples = torch.from_numpy(samples).float().to(device)
    parameters_ = samples @ parameters

    model_copy = copy.deepcopy(model)
    model_copy.eval()

    for param in parameters_:
        model_copy.vector_to_parameters(param)
        APL = model_copy.compute_APL(X_train, ccp_alpha)

        parameters_new.append(param)
        APLs_new.append(APL)

    del model_copy
    del parameters_
    del samples

    return parameters_new, APLs_new
