import numpy as np
from torch import nn
from sklearn.tree import DecisionTreeClassifier

def average_path_length(X_train, X_test, y_test, model: nn.Module):
    #y_tree = model.predict(X).detach().numpy()
    #y_tree = sequence_to_samples(y_tree)
    model.eval()
    y = model(X_train)
    # y_tree = np.argmax(y_tree, axis=1)
    #
    # X_tree = X.detach().numpy()
    # X_tree = sequence_to_samples(X_tree)

    """What is the correct way to create a pruned tree?
    If min_samples_leaf would be a float, this would reflect also the total numbers of samples.
    Otherwise, the trees could get more complex with bigger datasets."""
    tree = DecisionTreeClassifier(min_samples_leaf=25, ccp_alpha=0.001)
    X_train = X_train.to('cpu').detach().numpy()
    y = y.to('cpu').detach().numpy()
    y = [1 if y[i] > 0.5 else 0 for i in range(len(y))]
    tree.fit(X_train, y)
    tree = post_pruning(X_train, y, X_test, y_test, tree)
    return average_tree_length(X_train, tree)


def sequence_to_samples(tensor):
    sequence_array = [tensor[idx, :, :] for idx in range(tensor.shape[0])]
    return np.vstack(sequence_array)


def post_pruning(X_train, y_train,  X_test, y_test, tree):
    # https://medium.com/swlh/post-pruning-decision-trees-using-python-b5d4bcda8e23
    # https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html#sphx-glr-auto-examples-tree-plot-cost-complexity-pruning-py

    path = tree.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)

    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]

    index_best_model = np.argmax(test_scores)
    best_model = clfs[index_best_model]

    return best_model


def average_tree_length(X, tree):
    """
    The way the average tree length is calculated is different from the reference implementation
    at https://github.com/dtak/tree-regularization-public, but this seems to be the correct
    way.
    """
    path_length = np.mean(np.sum(tree.tree_.decision_path(X), axis=1))
    return path_length