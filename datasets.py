import numpy as np
from scipy.stats import bernoulli
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


def parabola(x): # parabola
    return 5 * (x - 0.75) ** 2 + 0.4


def polynom_3(x):
    return 7.75 * x ** 3 - 14 * x ** 2 + 6.25 * x + 0.25


def sample_2D_data(num_samples):
    samples = np.random.uniform(low=0, high=1.5, size=(num_samples, 2))
    labels = np.zeros(num_samples) # one-hot encoded

    for i in range(num_samples):
        x, y = samples[i, 0], samples[i, 1]
        if y > polynom_3(x):
            labels[i] = 1

        if np.abs(y - polynom_3(x)) < 0.15:
            labels[i] = bernoulli.rvs(0.5, 0.5)

    label_colors = ['r' if labels[i] == 1 else 'b' for i in range(labels)]

    return samples, labels, label_colors
