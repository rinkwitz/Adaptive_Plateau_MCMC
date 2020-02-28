import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import *

def unif_distr_exp_decaying_tails_pdf(y, mu, delta, sigma_1, sigma_2):
    C = np.sqrt(2 * np.pi * sigma_1 ** 2) / 2 + \
        np.sqrt(2 * np.pi * sigma_2 ** 2) / 2 + \
        2 * delta
    if y < mu - delta:
        return np.exp(-(y - (mu - delta)) ** 2 / (2 * sigma_1 ** 2)) / C
    elif mu - delta <= y and y <= delta + mu:
        return 1 / C
    else:
        return np.exp(-(y - (mu + delta)) ** 2 / (2 * sigma_2 ** 2)) / C

def trial_proposal(y, x, j, M, delta=2., delta_1=2., sigma=.05, sigma_0=3., sigma_1=3.):
    # k not specified
    if j == 0:
        return unif_distr_exp_decaying_tails_pdf(y, x, delta_1, sigma, sigma)
    elif j < M - 1:
        return (unif_distr_exp_decaying_tails_pdf(y, x - delta_1 - delta, delta, sigma, sigma) + \
                unif_distr_exp_decaying_tails_pdf(y, x + delta_1 + delta, delta, sigma, sigma)) / 2
    else:
        return (unif_distr_exp_decaying_tails_pdf(y, x - delta_1 - delta, delta, sigma_0, sigma) + \
                unif_distr_exp_decaying_tails_pdf(y, x + delta_1 + delta, delta, sigma, sigma_1)) / 2

def rejection_sampling_trial(x, j, M):
    while True:
        x2 = np.random.uniform()
        y2 = np.random.uniform()
        if trial_proposal(x2, x, j, M) < y2:
            return x2