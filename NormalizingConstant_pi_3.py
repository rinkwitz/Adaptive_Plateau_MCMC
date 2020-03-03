import numpy as np
from scipy.integrate import dblquad


def integrand_3(x_1, x_2):
    x = np.array([[x_1], [x_2]])
    A = np.array([[1, 1], [1, 1.5]])
    return (np.exp(-np.dot(np.dot(x.T, A), x) - np.cos(x[0, 0] / .1) - .5 * np.cos(x[1, 0] / .1)))[0, 0]


pi_3_normalizing_const = 1 / dblquad(lambda x_2, x_1: integrand_3(x_1, x_2), -np.inf, np.inf, lambda x_1: -np.inf,
                                     lambda x_1: np.inf)[0]