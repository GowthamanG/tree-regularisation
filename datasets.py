import numpy as np
from scipy.stats import bernoulli
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


def parabola(x): # parabola
    return 5 * (x - 0.5) ** 2 + 0.4


def polynom_3(x):
    return 2* x ** 3 - 4.8 * x ** 2 + 2.9 * x + 0.4
    #return x ** 3 - 3.6 * x ** 2 + 3.3 * x + 0.8


def polynom_6(x):
     return -2.19 * x ** 6 + 17.74 * x ** 5 - 54.24 * x ** 4 + 77.10 * x ** 3 - 49.66 * x ** 2 + 11.23 * x + 0.8


def sample_2D_data(num_samples, fun, space):
    vertical_dist_from_fun = lambda x, y: np.abs(y - fun(x))
    xx, yy = np.meshgrid(np.linspace(0, space[0], 100), np.linspace(0, space[1], 100))
    # Die Idee: Wahrscheinlichkeit für das Umdrehen für die jeweilige klasse ist größer,
    # je weiter man weg ist von der Trennlinie (by Fabricio)
    max_val = np.max(vertical_dist_from_fun(xx, yy))
    max_flip_prob = 0.5
    sharpness = 6
    flip_prob = lambda xx, yy: max_flip_prob * (1 - vertical_dist_from_fun(xx, yy) / max_val) ** 5
    class_ = lambda x, y: np.sign(y - fun(x))

    X = np.random.uniform(low=0, high=space[0], size=(num_samples, 2))
    Y = class_(X[:, 0], X[:, 1])
    do_flip = np.random.binomial(n=1, p=flip_prob(*X.T))
    Y = -(do_flip * 2 - 1) * Y
    Y = np.where(Y == -1, 0, Y)

    return X, Y

