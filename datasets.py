import numpy as np
from scipy.stats import bernoulli
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


def parabola(x): # parabola
    return 5 * (x - 0.75) ** 2 + 0.4


def polynom_3(x):
    return 2.8* x ** 3 - 6.7 * x ** 2 + 4 * x + 0.4


def sample_2D_data(num_samples, fun, space):
    vertical_dist_from_fun = lambda x, y: np.abs(y - fun(x))
    xx, yy = np.meshgrid(np.linspace(0, space[0], 100), np.linspace(0, space[1], 100))
    # Die Idee: Wahrscheinlichkeit für das Umdrehen für die jeweilige klasse ist größer,
    # je weiter man weg ist von der Trennlinie (by Fabricio)
    max_val = np.max(vertical_dist_from_fun(xx, yy))
    max_flip_prob = 0.5
    sharpness = 6
    flip_prob = lambda xx, yy: max_flip_prob * (1 - vertical_dist_from_fun(xx, yy) / max_val) ** 5
    class_ = lambda x, y: np.sign(polynom_3(x) - y)

    X = np.random.uniform(low=0, high=space[0], size=(num_samples, 2))
    Y = class_(X[:, 0], X[:, 1])
    do_flip = np.random.binomial(n=1, p=flip_prob(*X.T))
    Y = -(do_flip * 2 - 1) * Y

    colormap = {-1: "red", 1: "blue"}
    colormap = lambda Y: ['r' if i == -1 else 'b' for i in Y]

    # fig, ax = plt.subplots()
    # plt.xlim([0, 1.5])
    # plt.ylim([0, 1.5])
    # plt.scatter(samples[:,0], samples[:,1], c=colormap(Y))
    # ax.set_title('Training data')
    #
    # xx = np.linspace(0, 1.5, 50)
    # yy = polynom_3(xx)
    #
    # ax.plot(xx, yy, 'k-')
    #
    # plt.show()

    return X, Y

