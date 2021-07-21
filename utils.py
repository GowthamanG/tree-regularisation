import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from decision_tree_utils import average_path_length, post_pruning
from torch.utils.data import DataLoader, TensorDataset
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
from PIL import Image as ImagePIL
import pydotplus


def get_data_loader(X_train, y_train, X_test, y_test, batch_size, X_val=None, y_val=None):
    X_train = torch.tensor(X_train, dtype=torch.float)
    y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float)
    X_test = torch.tensor(X_test, dtype=torch.float)
    y_test = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float)
    #X_val = torch.tensor(X_val, dtype=torch.float)
    #y_val = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float)

    data_train = TensorDataset(X_train, y_train)
    data_train_loader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
    data_test = TensorDataset(X_test, y_test)
    data_test_loader = DataLoader(dataset=data_test, batch_size=batch_size)
    #data_val = TensorDataset(X_val, y_val)
    #data_val_loader = DataLoader(dataset=data_val, batch_size=batch_size)

    return data_train_loader, data_test_loader #, data_val_loader


def save_data(X, Y, filename: str):
    file_data = open(filename + '.txt', 'w')
    file_train_data = open(filename + '_train.txt', 'w')
    file_test_data = open(filename + '_test.txt', 'w')

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    np.savetxt(file_data, np.hstack((X, Y.reshape(-1, 1))))
    np.savetxt(file_train_data, np.hstack((X_train, y_train.reshape(-1, 1))))
    np.savetxt(file_test_data, np.hstack((X_test, y_test.reshape(-1, 1))))

    file_data.close()
    file_train_data.close()
    file_test_data.close()


def colormap(Y):
    return ['b' if y == 1 else 'r' for y in Y]


def build_decision_tree(X, y, X_train, y_train, X_test, y_test, space, path, ccp_alpha=None):

    if ccp_alpha:
        final_decision_tree = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
        final_decision_tree.fit(X_train, y_train)
    else:
        #final_decision_tree, ccp_alpha = post_pruning(X_train, y_train_predicted, X_test, y_test, final_decision_tree)
        ccp_alpha = post_pruning(X, y, X_train, y_train, X_test, y_test)
        final_decision_tree = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
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

    xx, yy = np.meshgrid(np.linspace(space[0][0], space[0][1], 100),
                         np.linspace(space[0][0], space[0][1], 100))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = final_decision_tree.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    fig_contour = plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
    plt.scatter(*X_train.T, c=colormap(y_train), edgecolors='k')
    plt.title(f'DT Contourplot with alpha: {ccp_alpha}')
    plt.savefig(f'{path}_contourplot.png')

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


def augment_data(X_train, X_test, y_test, model, device, size, ccp_alpha):
    model_copy = copy.deepcopy(model)
    parameters = []
    APLs = []

    # input_dim_surrogate_model = model_copy.surrogate_model.get_parameter_vector().shape[0]

    model_copy.eval()
    for _ in range(size):

        for param in model_copy.feed_forward.parameters():
            param.data.requires_grad = False

            # todo: 0.1 - 0.3 times relativ zur absoluten Wert des Parameters
            param_augmented = np.random.normal(param.data.cpu().numpy(), 0.1 * np.abs(param.data.cpu().numpy()))
            # param_augmented = np.random.normal(param.data.cpu().numpy(), 0.1)
            param.data = torch.tensor(param_augmented, dtype=torch.float).float().to(device)

        # parameters.append(model_copy.get_parameter_vector()[:-input_dim_surrogate_model])
        parameters.append(model_copy.get_parameter_vector)
        APLs.append(average_path_length(X_train, X_test, y_test, model_copy, ccp_alpha))

    del model_copy

    return parameters, APLs
