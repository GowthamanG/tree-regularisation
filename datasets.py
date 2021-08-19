import argparse
import numpy as np
import torch
import os
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

np.random.seed(5555)

def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--sample',
                        required=True,
                        type=str,
                        help='Type in which data set to sample. Availables: parabola, polynom_6, breast_cancer, signal_noise_hmm')

    parser.add_argument('--sample_size',
                        required=False,
                        type=int,
                        default=2000,
                        help='Number of data instances per sampling. Default: 2000')

    parser.add_argument('--path',
                        required=True,
                        type=str,
                        help='Directory where the data should be stored.')

    return parser


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


def polynom_3(x):
    return 2 * x ** 3 - 4.8 * x ** 2 + 2.9 * x + 0.2


def polynom_6(x):
    return -0.21 * x ** 6 + 0.01 * x ** 5 + 1.96 * x ** 4 - 0.17 * x ** 3 - 4.54 * x ** 2 + 0.63 * x + 2.15


def sample_2D_data(num_samples, fun, space):

    samples = np.random.uniform(low=space[0][0], high=space[0][1], size=(num_samples, 2))
    labels = np.zeros(num_samples) # one-hot encoded

    for i in range(num_samples):
        x, y = samples[i, 0], samples[i, 1]
        if np.abs(y > fun(x)):
            labels[i] = 1

        if y > fun(x) and (y < fun(x - 0.125) or y < fun(x + 0.125)):
            labels[i] = np.random.binomial(n=1, p=0.8)

        if y < fun(x) and (y > fun(x - 0.125) or y > fun(x + 0.125)):
            labels[i] = np.random.binomial(n=1, p=0.2)

    return samples, labels.reshape(-1, 1)

# def sample_2D_data_2(num_samples):
#     fun = lambda x: 5 * (x - 0.5) ** 2 + 0.4
#     fun_lower = lambda x: 5 * (x - 0.5) ** 2 + 0.2
#     fun_upper = lambda x: 5 * (x - 0.5) ** 2 + 0.6
#
#     # vertical_dist_from_fun = lambda x, y: np.abs(y - fun(x))
#     class_ = lambda x, y: np.sign(fun(x) - y)
#
#     X = np.random.rand(num_samples, 2)
#     flip_prob = lambda x, y: (fun_lower(x) < y) * (y < fun_upper(x)) * 0.2
#     do_flip = -np.random.binomial(n=1, p=flip_prob(*X.T)) * 2 + 1
#     Y = class_(*X.T) * do_flip
#     Y = np.where(Y == 1., Y, 0)
#
#     return X, Y


def gen_synthetic_dataset(data_count, time_count):
    """Signal-and-Noise HMM dataset
    Obtained from https://github.com/dtak/tree-regularization-public
    The generative process comes from two separate HMM processes. First,
    a "signal" HMM generates the first 7 data dimensions from 5 well-separated states.
    Second, an independent "noise" HMM generates the remaining 7 data dimensions
    from a different set of 5 states. Each timestep's output label is produced by a
    rule involving both the signal data and the signal hidden state.
    @param data_count: number of sequences in dataset
    @param time_count: number of timesteps in a sequence
    @return obs_set: Torch Tensor data_count x time_count x 14
    @return out_set: Torch Tensor data_count x time_count x 1
    """

    bias_mat = np.array([15])
    # 5 states + 7 observations
    weight_mat = np.array([[10, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0]])

    state_count = 5
    dim_count = 7
    out_count = 1

    # signal HMM process
    pi_mat_signal = np.array([.5, .5, 0, 0, 0])
    trans_mat_signal = np.array(([.7, .3, 0, 0, 0],
                                 [.5, .25, .25, 0, 0],
                                 [0, .25, .5, .25, 0],
                                 [0, 0, .25, .25, .5],
                                 [0, 0, 0, .5, .5]))
    obs_mat_signal = np.array(([.5, .5, .5, .5, 0, 0, 0],
                               [.5, .5, .5, .5, .5, 0, 0],
                               [.5, .5, .5, 0, .5, 0, 0],
                               [.5, .5, .5, 0, 0, .5, 0],
                               [.5, .5, .5, 0, 0, 0, .5]))

    # noise HMM process
    pi_mat_noise = np.array([.2, .2, .2, .2, .2])
    trans_mat_noise = np.array(([.2, .2, .2, .2, .2],
                                [.2, .2, .2, .2, .2],
                                [.2, .2, .2, .2, .2],
                                [.2, .2, .2, .2, .2],
                                [.2, .2, .2, .2, .2]))
    obs_mat_noise = np.array(([.5, .5, .5, 0, 0, 0, 0],
                              [0, .5, .5, .5, 0, 0, 0],
                              [0, 0, .5, .5, .5, 0, 0],
                              [0, 0, 0, .5, .5, .5, 0],
                              [0, 0, 0, 0, .5, .5, .5]))

    # create the sequences
    obs_set = np.zeros((dim_count * 2, time_count, data_count))
    out_set = np.zeros((out_count, time_count, data_count))

    state_set_signal = np.zeros((state_count, time_count, data_count))
    state_set_noise = np.zeros((state_count, time_count, data_count))

    # loop through to sample HMM states
    for data_ix in range(data_count):
        for time_ix in range(time_count):
            if time_ix == 0:
                state_signal = np.random.multinomial(1, pi_mat_signal)
                state_noise = np.random.multinomial(1, pi_mat_noise)
                state_set_signal[:, 0, data_ix] = state_signal
                state_set_noise[:, 0, data_ix] = state_noise
            else:
                tvec_signal = np.dot(state_set_signal[:, time_ix - 1, data_ix], trans_mat_signal)
                tvec_noise = np.dot(state_set_noise[:, time_ix - 1, data_ix], trans_mat_noise)
                state_signal = np.random.multinomial(1, tvec_signal)
                state_noise = np.random.multinomial(1, tvec_noise)
                state_set_signal[:, time_ix, data_ix] = state_signal
                state_set_noise[:, time_ix, data_ix] = state_noise

    # loop through to generate observations and outputs
    for data_ix in range(data_count):
        for time_ix in range(time_count):
            obs_vec_signal = np.dot(state_set_signal[:, time_ix, data_ix], obs_mat_signal)
            obs_vec_noise = np.dot(state_set_noise[:, time_ix, data_ix], obs_mat_noise)
            obs_signal = np.random.binomial(1, obs_vec_signal)
            obs_noise = np.random.binomial(1, obs_vec_noise)
            obs = np.hstack((obs_signal, obs_noise))  # concat together
            obs_set[:, time_ix, data_ix] = obs

            # input is state concatenated with observation
            in_vec = np.hstack((state_set_signal[:, time_ix, data_ix],
                                obs_set[:dim_count, time_ix, data_ix]))

            # output is a logistic regression on W \dot input
            out_vec = 1 / (1 + np.exp(-1 * (np.dot(weight_mat, in_vec) - bias_mat)))

            out = np.random.binomial(1, out_vec)
            out_set[:, time_ix, data_ix] = out

    return obs_set.T, out_set.T


if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else 'cpu'

    args = parser().parse_args()
    num_samples = args.sample_size

    if args.sample == 'parabola':
        dim = 2
        space = [[0, 1.5], [0, 1.5]]
        fun_name = 'parabola'
        X, Y = sample_2D_data(num_samples, parabola, space)
        save_data(X, Y, f'{args.path}/data_{fun_name}')

    elif args.sample == 'polynom_6':
        dim = 2
        space = [[-3, 3], [-3, 3]]
        fun_name = 'polynom_6'
        X, Y = sample_2D_data(num_samples, polynom_6, space)
        save_data(X, Y, f'{args.path}/data_{fun_name}')

    elif args.sample == 'breast_cancer':
        path = f'{args.path}'
        if not os.path.exists(path):
            os.makedirs(path)

        data = load_breast_cancer()
        num_samples = data.data.shape[0]
        dim = data.data.shape[1]
        save_data(data.data, data.target, f'{args.path}/data_{args.sample}')

    elif args.sample == 'signal_noise_hmm':
        path = f'{args.path}'
        if not os.path.exists(path):
            os.makedirs(path)

        time_count = 50
        X, y = gen_synthetic_dataset(num_samples, time_count)
        save_data(X, y, f'{args.path}/data_{args.sample}')
