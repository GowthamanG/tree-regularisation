import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils import *

np.random.seed(5555)


def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--sample',
                        required=True,
                        type=str,
                        help='Available functions to sample from: parabola, cos')

    parser.add_argument('--sample_size',
                        required=False,
                        type=int,
                        default=2000,
                        help='Number of data instances per sampling. Default: 2000')

    parser.add_argument('--path',
                        required=True,
                        type=str,
                        help='Directory, where the data should be stored.')

    return parser


def plot(X, y, fun, error, space):

    x_lower = lambda x: fun(x) - error
    x_upper = lambda x: fun(x) + error

    x_decision_fun = np.linspace(space[0][0], space[0][1], 100)
    y_decision_fun = fun(x_decision_fun)

    fig = plt.figure()
    plt.scatter(*X.T, c=colormap(y), edgecolors='k')
    plt.xlim([space[0][0], space[0][1]])
    plt.ylim([space[1][0], space[1][1]])
    plt.title('Samples')
    plt.plot(x_decision_fun, y_decision_fun, 'k-')
    plt.plot(x_decision_fun, x_lower(x_decision_fun), color='#808080')
    plt.plot(x_decision_fun, x_upper(x_decision_fun), color='#808080')
    plt.show()
    plt.close(fig)

    fig = plt.figure()
    plt.xlim([space[0][0], space[0][1]])
    plt.ylim([space[1][0], space[1][1]])
    plt.title('Samples')
    plt.plot(x_decision_fun, y_decision_fun, 'k-')
    plt.plot(x_decision_fun, x_lower(x_decision_fun), color='#808080')
    plt.plot(x_decision_fun, x_upper(x_decision_fun), color='#808080')
    plt.fill_between(x_decision_fun, x_lower(x_decision_fun), x_upper(x_decision_fun))
    plt.show()
    plt.close(fig)


def save_data(X, y, filename: str):

    file_data = open(filename + '.txt', 'w')
    file_train_data = open(filename + '_train.txt', 'w')
    file_test_data = open(filename + '_test.txt', 'w')
    file_val_data = open(filename + '_val.txt', 'w')

    y = y.reshape(-1, 1)

    # data split 70/15/15 ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    np.savetxt(file_data, np.hstack((X, y)))
    np.savetxt(file_train_data, np.hstack((X_train, y_train)))
    np.savetxt(file_test_data, np.hstack((X_test, y_test)))
    np.savetxt(file_val_data, np.hstack((X_val, y_val)))

    file_data.close()
    file_train_data.close()
    file_test_data.close()
    file_val_data.close()


def parabola(x):
    return 5 * (x - 0.75) ** 2 + 0.4


def cos(x):
    return np.cos(x)


def sample_2D_data(num_samples, fun, error, space):
    samples = np.random.uniform(low=space[0][0], high=space[0][1], size=(num_samples, 2))
    labels = np.where(samples[:, 1] > fun(samples[:, 0]), 1, 0)

    fun_lower = lambda x: fun(x) - error
    fun_upper = lambda x: fun(x) + error

    for i, (x, y) in enumerate(samples):
        if fun_lower(x) <= y <= fun_upper(x):
            labels[i] = np.random.binomial(n=1, p=0.5)

    return samples, labels


if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else 'cpu'

    args = parser().parse_args()
    num_samples = args.sample_size

    if args.sample == 'parabola':
        dim = 2
        space = [[0, 1.5], [0, 1.5]]
        fun_name = 'parabola'
        X, Y = sample_2D_data(num_samples, parabola, 0.2, space)
        plot(X, Y, parabola, 0.2, space)
        save_data(X, Y, f'{args.path}/data_{fun_name}')

    elif args.sample == 'cos':
        dim = 2
        space = [[-6, 6], [-2, 2]]
        fun_name = 'cos'
        X, Y = sample_2D_data(num_samples, cos, 0.4, space)
        plot(X, Y, cos, 0.4, space)
        save_data(X, Y, f'{args.path}/data_{fun_name}')
