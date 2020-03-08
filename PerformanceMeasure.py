import numpy as np


def ACT(simulations):
    num_simulations = simulations.shape[0]
    num_components = simulations.shape[2]
    act = np.empty((num_simulations, num_components))
    for i in range(num_simulations):
        for k in range(num_components):
            N = len(simulations[i, :, k])
            vec = simulations[i, :, k].reshape((-1, 1))
            cov = 1 / (N - 1) * np.dot(vec - np.mean(vec), vec.T - np.mean(vec))
            var = 1 / (N - 1) * np.sum((vec - np.mean(vec)) ** 2)
            act[i, k] = 1 + 2 / var * np.sum(cov[0, 1:])
    print('median act component-wise:', np.median(act, axis=0))
    return act


def ASJD(simulations):
    num_simulations = simulations.shape[0]
    num_components = simulations.shape[2]
    asjd = np.empty((num_simulations, num_components))
    for i in range(num_simulations):
        for k in range(num_components):
            asjd[i, k] = np.mean((simulations[i, 1:, k] - simulations[i, :-1, k]) ** 2)
    print('median asjd component-wise:', np.median(asjd, axis=0))
    return asjd
