import numpy as np
from torch import nn
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def average_path_length(X_train, X_test, y_test, model: nn.Module):
    model.eval()
    y_tree = model(X_train).to('cpu').detach().numpy()
    #y_tree = sequence_to_samples(y_tree)
    #y_tree = np.argmax(y_tree, axis=1)

    X_tree = X_train.to('cpu').detach().numpy()
    #X_tree = sequence_to_samples(X_tree)

    """What is the correct way to create a pruned tree?
    If min_samples_leaf would be a float, this would reflect also the total numbers of samples.
    Otherwise, the trees could get more complex with bigger datasets."""
    tree = DecisionTreeClassifier(random_state=42, min_samples_leaf=25)

    y_tree = np.where(y_tree > 0.5, 1, -1)
    tree.fit(X_tree, y_tree)
    #tree = post_pruning(X_tree, y_tree, X_test.to('cpu').detach().numpy(), y_test.to('cpu').detach().numpy(), tree)
    return average_tree_length(X_tree, tree)


def sequence_to_samples(tensor):
    sequence_array = [tensor[idx, :, :] for idx in range(tensor.shape[0])]
    return np.vstack(sequence_array)


def post_pruning(X_train, y_train, X_test, y_test, tree):
    # https://medium.com/swlh/post-pruning-decision-trees-using-python-b5d4bcda8e23
    # https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html#sphx-glr-auto-examples-tree-plot-cost-complexity-pruning-py

    path = tree.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)

    # remove last tree since it is a trivial one with only one node
    clfs = clfs[:-1]
    ccp_alphas = ccp_alphas[:-1]

    #train_scores = [clf.score(X_train, y_train) for clf in clfs]
    #test_scores = [clf.score(X_test, y_test) for clf in clfs]

    test_scores = [accuracy_score(y_test, clf.predict(X_test)) for clf in clfs]

    # Select the alpha with max test accuracy
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
