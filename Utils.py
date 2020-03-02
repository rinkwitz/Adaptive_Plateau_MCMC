import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import *

def unif_distr_exp_decaying_tails_pdf(y, mu, delta, sigma_1, sigma_2):
    C = np.sqrt(2 * np.pi * sigma_1 ** 2) / 2 + np.sqrt(2 * np.pi * sigma_2 ** 2) / 2 + 2 * delta
    if y < mu - delta:
        return np.exp(-(y - (mu - delta)) ** 2 / (2 * sigma_1 ** 2)) / C
    elif mu - delta <= y and y <= delta + mu:
        return 1 / C
    else:
        return np.exp(-(y - (mu + delta)) ** 2 / (2 * sigma_2 ** 2)) / C

def trial_proposal(x, y, j, M, delta=2., delta_1=2., sigma=.05, sigma_0=3., sigma_1=3.):
    # k not specified
    if j == 0:
        return unif_distr_exp_decaying_tails_pdf(y, x, delta_1, sigma, sigma)
    elif j < M - 1:
        return (unif_distr_exp_decaying_tails_pdf(y, x - (2 * j - 1) * delta - delta_1, delta, sigma, sigma) + unif_distr_exp_decaying_tails_pdf(y, x + (2 * j - 1) * delta + delta_1, delta, sigma, sigma)) / 2
    else:
        return (unif_distr_exp_decaying_tails_pdf(y, x - (2 * j - 1) * delta - delta_1, delta, sigma_0, sigma) + unif_distr_exp_decaying_tails_pdf(y, x + (2 * j - 1) * delta + delta_1, delta, sigma, sigma_1)) / 2

def rejection_sampling_trial(x, j, M):
    # maybe we have to make this adaptive ...
    if j == 0:
        g = 1 / 6
        c = (1 / np.sqrt(2 * np.pi * .05 ** 2) / 2 + np.sqrt(2 * np.pi * .05 ** 2) / 2 + 2 * 2) / g
    elif j < M - 1:
        g = 1 / 6
        c = (1 / np.sqrt(2 * np.pi * .05 ** 2) / 2 + np.sqrt(2 * np.pi * .05 ** 2) / 2 + 2 * 2) / (2 * g)
    else:
        g = 1 / 15
        c = (1 / np.sqrt(2 * np.pi * 3 ** 2) / 2 + np.sqrt(2 * np.pi * .05 ** 2) / 2 + 2 * 2) / (2 * g)
    while True:
        if j == 0:
            Y = np.random.uniform(x - 3, x + 3)
        elif j < M - 1:
            if np.random.rand() < .5:
                Y = np.random.uniform(x - (2 * j - 1) * 2 - 5, x - (2 * j - 1) * 2 + 1)
            else:
                Y = np.random.uniform(x + (2 * j - 1) * 2 - 1, x + (2 * j - 1) * 2 + 5)
        else:
            if np.random.rand() < .5:
                Y = np.random.uniform(x - (2 * j - 1) * 2 - 12, x - (2 * j - 1) * 2 + 1)
            else:
                Y = np.random.uniform(x + (2 * j - 1) * 2 - 1, x + (2 * j - 1) * 2 + 12)
        U = np.random.uniform()
        if U < trial_proposal(x, Y, j, M) / (c * g):
            return Y

def trial_weight(z, x, k, j, M):
    x_replacement = x.copy()
    x_replacement[k] = z
    # define target distr pi here:
    # pi_1: mixture of gaussians
    mu_1 = np.array([5, 5, 0, 0])
    mu_2 = np.array([15, 15, 0, 0])
    Sigma_1 = np.diag([6.25, 6.25, 6.25, 0.01])
    Sigma_2 = np.diag([6.25, 6.25, .25, 0.01])
    pi = multivariate_normal(mean=mu_1, cov=Sigma_1).pdf(x_replacement) / 2 + multivariate_normal(mean=mu_2, cov=Sigma_2).pdf(x_replacement) / 2
    return pi * trial_proposal(x[k], z, j, M) * lambda_function(x[k], z, j, M)

def lambda_function(x, y, j, M):
    alpha = 2.5
    return trial_proposal(x, y, j, M) * np.abs(x - y) ** alpha

def draw_from_z_proportional_to_w(z, w):
    if np.sum(w) != 0:
        ps = np.array(w) / np.sum(w)
        u = np.random.rand()
        p_cumul = 0.0
        for i, p in enumerate(ps):
            p_cumul += p
            if u <= p_cumul:
                return z[i]
    else:
        return z[0]