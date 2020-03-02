import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import *

def unif_distr_exp_decaying_tails_pdf(y, mu, delta, sigma_1, sigma_2):
    C = np.sqrt(2 * np.pi * sigma_1 ** 2) / 2 + np.sqrt(2 * np.pi * sigma_2 ** 2) / 2. + 2 * delta
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
        return (unif_distr_exp_decaying_tails_pdf(y, x - j * (delta_1 + delta), delta, sigma, sigma) + unif_distr_exp_decaying_tails_pdf(y, x + j * (delta_1 + delta), delta, sigma, sigma)) / 2
    else:
        return (unif_distr_exp_decaying_tails_pdf(y, x - j * (delta_1 + delta), delta, sigma_0, sigma) + unif_distr_exp_decaying_tails_pdf(y, x + j * (delta_1 + delta), delta, sigma, sigma_1)) / 2

def rejection_sampling_trial(x, j, M):
    if j == 0:
        upper_bound = np.max([trial_proposal(p, x, j, M)/norm(loc=x, scale=2.).pdf(p) for p in np.linspace(x - 4, x + 4, 100)])
        while True:
            Y = norm(loc=x, scale=2.).rvs()
            U = np.random.uniform()
            if U < trial_proposal(Y, x, j, M) / (upper_bound * norm(loc=x, scale=2.).pdf(Y)):
                return Y
    elif j < M - 1:
        upper_bound = np.max(
            [trial_proposal(p, x, j, M) / (norm(loc=x, scale=2.).pdf(p) / 2.) for p in np.linspace(x - (j+1)*4, x - (j-1) * 4, 100)])
        while True:
            U = np.random.uniform()
            if np.random.rand() < .5:
                Y = norm(loc=x - 4 * j, scale=2.).rvs()
                if U < trial_proposal(Y, x, j, M) / (upper_bound * norm(loc=x - 4 * j, scale=2.).pdf(Y)):
                    return Y
            else:
                Y = norm(loc=x+4*j, scale=2.).rvs()
                if U < trial_proposal(Y, x, j, M) / (upper_bound * norm(loc=x + 4 * j, scale=2.).pdf(Y)):
                    return Y
    else:
        upper_bound = (1 / (np.sqrt(2 * np.pi * .05 ** 2) / 2 + np.sqrt(2 * np.pi * 3. ** 2) / 2. + 2 * 2.)) / 2.
    #while True:
    #    Y = np.random.uniform(x - j * 4, x + j * 4)
    #    U = np.random.uniform()
    #    if U < trial_proposal(Y, x, j, M) / (upper_bound ):
    #        return Y
        #print(Y, x, j, M, trial_proposal(Y, x, j, M))