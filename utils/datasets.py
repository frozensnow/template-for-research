import os
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import torch
from torch.utils.data import Dataset, DataLoader


def make_var_stationary(beta, radius=0.97):
    '''Rescale coefficients of VAR model to make stable.'''
    p = beta.shape[0]
    lag = beta.shape[1] // p
    bottom = np.hstack((np.eye(p * (lag - 1)), np.zeros((p * (lag - 1), p))))
    beta_tilde = np.vstack((beta, bottom))
    eigvals = np.linalg.eigvals(beta_tilde)
    max_eig = max(np.abs(eigvals))
    nonstationary = max_eig > radius
    if nonstationary:
        return make_var_stationary(0.95 * beta, radius)
    else:
        return beta


def simulate_var(p, T, lag, gc=None, sparsity=0.2, beta_value=1.0, sd=0.1, seed=0):
    if seed is not None:
        np.random.seed(seed)

    if gc is None:
        # Set up coefficients and Granger causality ground truth.
        GC = np.eye(p, dtype=int)
        beta = np.eye(p) * beta_value

        num_nonzero = int(p * sparsity) - 1
        for i in range(p):
            choice = np.random.choice(p - 1, size=num_nonzero, replace=False)
            choice[choice >= i] += 1
            beta[i, choice] = beta_value
            GC[i, choice] = 1
    else:
        GC = gc.astype(int)
        beta = GC * beta_value

    beta = np.hstack([beta for _ in range(lag)])
    beta = make_var_stationary(beta)

    # Generate data.
    burn_in = 100
    errors = np.random.normal(scale=sd, size=(p, T + burn_in))
    X = np.zeros((p, T + burn_in))
    X[:, :lag] = errors[:, :lag]
    for t in range(lag, T + burn_in):
        X[:, t] = np.dot(beta, X[:, (t-lag):t].flatten(order='F'))
        X[:, t] += + errors[:, t-1]

    return X.T[burn_in:], beta, GC


def lorenz(x, t, F):
    '''Partial derivatives for Lorenz-96 ODE.'''
    p = len(x)
    dxdt = np.zeros(p)
    for i in range(p):
        dxdt[i] = (x[(i+1) % p] - x[(i-2) % p]) * x[(i-1) % p] - x[i] + F

    return dxdt


def simulate_lorenz_96(p, T, F=10.0, delta_t=0.1, sd=0.1, burn_in=1000,
                       seed=0):
    if seed is not None:
        np.random.seed(seed)

    # Use scipy to solve ODE.
    x0 = np.random.normal(scale=0.01, size=p)
    t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)
    X = odeint(lorenz, x0, t, args=(F,))
    X += np.random.normal(scale=sd, size=(T + burn_in, p))

    # Set up Granger causality ground truth.
    GC = np.zeros((p, p), dtype=int)
    for i in range(p):
        GC[i, i] = 1
        GC[i, (i + 1) % p] = 1
        GC[i, (i - 1) % p] = 1
        GC[i, (i - 2) % p] = 1

    return X[burn_in:], GC


# nci special
def tvc_set(raw_series, list_type):
    """
    Temporal Variation Characterization: data preprocess
    :param raw_series: array{len_series x num_series}
    :param list_type: list{num_series}
    :return: array{num_series x len_train_series(len_series - 3) x 3}
    """
    character_var = []
    for i, var_type in enumerate(list_type):
        if var_type == 0:
            tmp1 = np.diff(raw_series[:, i:i+1], n=1, axis=0)[2:]
            tmp2 = np.diff(raw_series[:, i:i+1], n=2, axis=0)[1:]
            tmp3 = np.diff(raw_series[:, i:i+1], n=3, axis=0)
            character_var.append(
                np.hstack((tmp1, tmp2, tmp3))
            )
        else:
            tmp = np.diff(raw_series[:, i:i+1], n=1, axis=0)[2:]
            temp = np.zeros((len(tmp), 3))
            for j in range(len(tmp)):
                if tmp[j] == 0:
                    temp[j, 0] = 1
                elif tmp[j] > 0:
                    temp[j, 1] = tmp[j]
                elif tmp[j] < 0:
                    temp[j, 2] = -tmp[j]
            character_var.append(temp)

    return np.array(character_var).transpose(1, 0, 2)


class DatasetVAR(Dataset):
    def __init__(self, root_path, data_path, list_type, flag='train', lag=10):
        # init
        assert flag in ['train', 'vali', 'test']
        type_map = {'train': 0, 'vali': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.data_path = data_path
        self.lag = lag
        self.list_type = list_type
        self.__read_data__()

    def __read_data__(self):
        raw_data = np.load(os.path.join(self.root_path, self.data_path))
        data = raw_data['arr_0']
        self.variate_ch = tvc_set(data, self.list_type)
        self.data = data[3:, :]
        self.GC = raw_data['arr_1']
        self.seq_len = self.data.shape[0]
        self.num_series = self.data.shape[1]

    def __getitem__(self, index):

        x = torch.from_numpy(self.data[index:index + self.lag + 1])
        x_ = torch.from_numpy(self.variate_ch[index:index + self.lag, :, :])

        return x, x_

    def __len__(self):
        return len(self.data) - self.lag


if __name__ == '__main__':
    X_np, beta, GC = simulate_var(p=5, T=1000, lag=3, seed=2024, gc=np.array([[1, 0, 0, 0, 0],
                                                                              [0, 1, 0, 0, 0],
                                                                              [1, 1, 1, 0, 0],
                                                                              [1, 0, 1, 1, 0],
                                                                              [0, 0, 1, 0, 1]], dtype=int))
    np.savez('../data/var/var2.npz', X_np, beta, GC)
