import numpy as np

np.random.seed(5555)

def parabola(x): # parabola
    return 5 * (x - 0.75) ** 2 + 0.2


def polynom_3(x):

    return 2* x ** 3 - 4.8 * x ** 2 + 2.9 * x + 0.2
    #return x ** 3 - 3.6 * x ** 2 + 3.3 * x + 0.8


def polynom_6(x):
    #space => [-3,3] x [-3,3]
     #return -2.19 * x ** 6 + 17.74 * x ** 5 - 54.24 * x ** 4 + 77.10 * x ** 3 - 49.66 * x ** 2 + 11.23 * x + 0.8
    return -0.21 * x ** 6 + 0.01 * x ** 5 + 1.96 * x ** 4 - 0.17 * x ** 3 - 4.54 * x ** 2 + 0.63 * x + 2.15


def sample_2D_data(num_samples, fun, space):

    samples = np.random.uniform(low=space[0][0], high=space[0][1], size=(num_samples, 2))
    labels = np.zeros(num_samples) # one-hot encoded

    for i in range(num_samples):
        x, y = samples[i, 0], samples[i, 1]
        if np.abs(y > fun(x)):
            labels[i] = 1

        if y > fun(x) and (y < fun(x - 0.15) or y < fun(x + 0.15)):
            labels[i] = np.random.binomial(n=1, p=0.8)

        if y < fun(x) and (y > fun(x - 0.15) or y > fun(x + 0.15)):
            labels[i] = np.random.binomial(n=1, p=0.2)

    return samples, labels
